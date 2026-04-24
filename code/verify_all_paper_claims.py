#!/usr/bin/env python3
"""
Verify ALL paper claims against the current final_normalized_100q/ data.

This script covers everything NOT already in recompute_all_tables.py:
  - Table 1 (tab:model_summary): N, Q, Responses, Resp%, Per-Q Var
  - Table (tab:matrix_similarity): KL, JS, Pearson on flattened matrices
  - Appendix 8x8 pairwise KL, JS, Pearson matrices
  - Inline numbers: 1.8-3.6x variance ratio, 63.2% missing, etc.
  - Figure 1 caption numbers
  - Parsing success rates (from raw eval data)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


# Resolve data dir relative to this script: release/code/*.py → release/data/
import os

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
NORMALIZED_DIR = DATA_DIR / "final_normalized_100q"
MODEL_FILES = {
    "Human": "human_survey_normalized.json",
    "GPT-5.1": "gpt51_normalized.json",
    "GPT-4o": "openai_gpt4o_normalized.json",
    "Claude Sonnet 4.5": "sonnet45_normalized.json",
    "Llama 3.1 8B": "llama3p18b_normalized.json",
    "Llama 3.1 8B (FT)": "llama3p18b_finetuned_normalized.json",
    "Mistral 7B": "mistral7b_normalized.json",
    "Qwen 3 4B": "qwen3-4b_normalized.json",
}


def build_matrices():
    """Load all data and build aligned matrices."""
    all_data = {}
    for name, fname in MODEL_FILES.items():
        with open(NORMALIZED_DIR / fname) as f:
            all_data[name] = json.load(f)

    # Use human philosopher ordering
    phil_names = [p["name"] for p in all_data["Human"]]

    # Get question keys from human data
    q_keys = set()
    for p in all_data["Human"]:
        q_keys.update(k for k, v in p.get("responses", {}).items() if v is not None)
    q_keys = sorted(q_keys)

    matrices = {}
    for name, data in all_data.items():
        name2p = {p["name"]: p for p in data}
        mat = np.full((len(phil_names), len(q_keys)), np.nan)
        for i, nm in enumerate(phil_names):
            p = name2p.get(nm, {})
            for j, k in enumerate(q_keys):
                v = p.get("responses", {}).get(k)
                if v is not None:
                    mat[i, j] = v
        matrices[name] = mat

    return matrices, phil_names, q_keys, all_data


def kl_div(p_vals, q_vals, n_bins=20):
    bins = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    eps = 1e-10
    p_hist = (p_hist + eps) / (p_hist + eps).sum()
    q_hist = (q_hist + eps) / (q_hist + eps).sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def js_div(p_vals, q_vals, n_bins=20):
    bins = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    eps = 1e-10
    p_hist = (p_hist + eps) / (p_hist + eps).sum()
    q_hist = (q_hist + eps) / (q_hist + eps).sum()
    return float(jensenshannon(p_hist, q_hist) ** 2)


def main():
    matrices, phil_names, q_keys, all_data = build_matrices()

    print("=" * 80)
    print("COMPREHENSIVE PAPER CLAIM VERIFICATION")
    print("Data source: final_normalized_100q/ (B&C most-popular normalization)")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────
    # TABLE 1: Model Summary Statistics (tab:model_summary)
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 1 (tab:model_summary): N, Q, Responses, Resp%, Per-Q Var")
    print("=" * 80)
    print(f"  Paper claims: Human Per-Q Var=0.071, range 1.8-3.6x")
    print()
    print(
        f"  {'Model':<22} {'N':>5} {'Q':>5} {'Responses':>10} {'Resp%':>8} {'Per-Q Var':>10}"
    )
    print("  " + "-" * 62)

    human_var = None
    for name in MODEL_FILES:
        mat = matrices[name]
        n_phil = mat.shape[0]
        n_q = mat.shape[1]
        n_responses = int(np.sum(~np.isnan(mat)))
        resp_pct = n_responses / mat.size * 100
        # Per-Q variance
        q_vars = []
        for j in range(n_q):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 1:
                q_vars.append(np.var(valid))
        avg_var = np.mean(q_vars) if q_vars else 0
        if name == "Human":
            human_var = avg_var
        ratio = human_var / avg_var if human_var and avg_var > 0 else 0
        ratio_str = f"({ratio:.1f}x)" if name != "Human" else ""
        print(
            f"  {name:<22} {n_phil:>5} {n_q:>5} {n_responses:>10,} {resp_pct:>7.1f}% {avg_var:>10.4f} {ratio_str}"
        )

    # ──────────────────────────────────────────────────────────────────
    # TABLE (tab:matrix_similarity): Flattened response matrix similarity
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE (tab:matrix_similarity): KL, JS, Pearson on flattened matrices")
    print("=" * 80)
    print(f"  Paper claims: Sonnet JS=0.109 r=0.425, GPT-5.1 KL=1.289")
    print()
    print(f"  {'Model':<22} {'KL':>8} {'JS':>8} {'Pearson r':>10}")
    print("  " + "-" * 50)

    h = matrices["Human"]
    for name in MODEL_FILES:
        if name == "Human":
            continue
        m = matrices[name]
        mask = ~(np.isnan(h) | np.isnan(m))
        hv, mv = h[mask], m[mask]
        kl = kl_div(hv, mv)
        js = js_div(hv, mv)
        r, _ = pearsonr(hv, mv)
        print(f"  {name:<22} {kl:>8.3f} {js:>8.3f} {r:>10.3f}")

    # ──────────────────────────────────────────────────────────────────
    # INLINE NUMBERS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("INLINE NUMBERS VERIFICATION")
    print("=" * 80)

    # Missing data rates
    for name in MODEL_FILES:
        mat = matrices[name]
        miss = np.isnan(mat).sum() / mat.size * 100
        print(f"  {name:<22} missing: {miss:.1f}%")

    # Per-Q variance extremes for Figure 1 caption
    print()
    for name in ["Human", "Claude Sonnet 4.5", "Llama 3.1 8B"]:
        mat = matrices[name]
        q_vars = []
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 1:
                q_vars.append((np.var(valid), q_keys[j]))
        q_vars.sort(key=lambda x: x[0])
        zero_var = sum(1 for v, _ in q_vars if v < 0.001)
        avg = np.mean([v for v, _ in q_vars])
        print(f"  {name}: avg_var={avg:.4f}, zero_var_q={zero_var}")
        print(f"    Lowest: {q_vars[0][1]} (var={q_vars[0][0]:.4f})")
        print(f"    Highest: {q_vars[-1][1]} (var={q_vars[-1][0]:.4f})")

    # Inline: physicalism -> atheism correlation
    print()
    from recompute_all_tables import pairwise_corr_matrix

    human_mat = matrices["Human"]
    corr = pairwise_corr_matrix(human_mat)
    q_stems = {q_keys[j].split(":")[0].strip().lower(): j for j in range(len(q_keys))}
    mind_idx = q_stems.get("mind")
    god_idx = q_stems.get("god")
    if mind_idx is not None and god_idx is not None:
        print(f"  mind <-> god correlation: r = {corr[mind_idx, god_idx]:.3f}")
        print(f"  Paper claims: r ≈ 0.37")

    # ──────────────────────────────────────────────────────────────────
    # APPENDIX: 8x8 Pairwise Matrices
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("APPENDIX: 8x8 Pairwise KL Divergence (tab:kl_full)")
    print("=" * 80)
    model_names = list(MODEL_FILES.keys())

    # KL matrix
    header = f"  {'':>15}"
    for n in model_names:
        short = n[:8]
        header += f" {short:>8}"
    print(header)

    for ni in model_names:
        row = f"  {ni[:15]:>15}"
        for nj in model_names:
            if ni == nj:
                row += f" {'---':>8}"
            else:
                mi, mj = matrices[ni], matrices[nj]
                mask = ~(np.isnan(mi) | np.isnan(mj))
                if mask.sum() > 0:
                    kl = kl_div(mi[mask], mj[mask])
                    row += f" {kl:>8.3f}"
                else:
                    row += f" {'N/A':>8}"
        print(row)

    # JS matrix
    print("\n  8x8 JS Divergence:")
    for ni in model_names:
        row = f"  {ni[:15]:>15}"
        for nj in model_names:
            if ni == nj:
                row += f" {'---':>8}"
            else:
                mi, mj = matrices[ni], matrices[nj]
                mask = ~(np.isnan(mi) | np.isnan(mj))
                if mask.sum() > 0:
                    js = js_div(mi[mask], mj[mask])
                    row += f" {js:>8.3f}"
                else:
                    row += f" {'N/A':>8}"
        print(row)

    # Pearson matrix
    print("\n  8x8 Pearson Correlation:")
    for ni in model_names:
        row = f"  {ni[:15]:>15}"
        for nj in model_names:
            if ni == nj:
                row += f" {'---':>8}"
            else:
                mi, mj = matrices[ni], matrices[nj]
                mask = ~(np.isnan(mi) | np.isnan(mj))
                if mask.sum() > 10:
                    r, _ = pearsonr(mi[mask], mj[mask])
                    row += f" {r:>8.3f}"
                else:
                    row += f" {'N/A':>8}"
        print(row)

    # ──────────────────────────────────────────────────────────────────
    # DPO inline numbers
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DPO INLINE NUMBERS")
    print("=" * 80)

    base = matrices["Llama 3.1 8B"]
    ft = matrices["Llama 3.1 8B (FT)"]
    human = matrices["Human"]

    # Agnostic percentage (values close to 0.5)
    for label, mat in [("Base", base), ("FT", ft), ("Human", human)]:
        valid = mat[~np.isnan(mat)]
        agnostic = np.sum((valid >= 0.375) & (valid <= 0.625)) / len(valid) * 100
        print(f"  {label} Agnostic%: {agnostic:.1f}%")

    # GPT-5.1 vs GPT-4o similarity
    g51 = matrices["GPT-5.1"]
    g4o = matrices["GPT-4o"]
    mask = ~(np.isnan(g51) | np.isnan(g4o))
    js_gpt = js_div(g51[mask], g4o[mask])
    r_gpt, _ = pearsonr(g51[mask], g4o[mask])
    print(f"\n  GPT-5.1 vs GPT-4o: JS={js_gpt:.3f}, r={r_gpt:.3f}")
    print(f"  Paper claims: JS=0.008, r=0.932")


if __name__ == "__main__":
    main()
