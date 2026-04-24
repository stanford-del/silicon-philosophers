#!/usr/bin/env python3
"""
Recompute ALL paper tables from final_normalized_100q/ data (100 question stems).

This script produces every number in the paper that depends on response data:
  - Table 2: Demographic-Position Correlations
  - Table 3: Spurious specialist effects
  - Tables 4/5: Question-level and Domain-level predictability (RMSE)
  - Table domain_per_model: Per-model RMSE by domain
  - Table 6: PCA structural comparison
  - Table PC1 comparison: Top-5 PC1 loadings
  - Table 7: Question correlation structure (Mantel, RV, KL, JS)
  - Table 8: Fine-tuning trade-off (entropy, resp KL, corr KL/JS)
  - Appendix tables: detailed demographic, predictable questions,
    per-component loadings, fine-tuning by question
  - All inline numbers

Reads responses from: final_normalized_100q/
Reads demographics from: merged_*_philosophers_normalized.json (joined by name)
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.stats import pearsonr, chi2_contingency
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Resolve data dir relative to this script: release/code/*.py → release/data/
# Override with DATA_DIR env var if needed.
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

DEMOGRAPHIC_FILES = {
    "Human": "merged_human_survey_philosophers_normalized.json",
    "GPT-5.1": "merged_gpt51_philosophers_normalized.json",
    "GPT-4o": "merged_openai_gpt4o_philosophers_normalized.json",
    "Claude Sonnet 4.5": "merged_sonnet45_philosophers_normalized.json",
    "Llama 3.1 8B": "merged_llama3p18b_philosophers_normalized.json",
    "Llama 3.1 8B (FT)": "merged_llama3p18b_finetuned_philosophers_normalized.json",
    "Mistral 7B": "merged_mistral7b_philosophers_normalized.json",
    "Qwen 3 4B": "merged_qwen3-4b_philosophers_normalized.json",
}

# Domain assignments for 100 questions (base question stems)
DOMAIN_MAP = {
    "a priori knowledge": "Epistemology",
    "abortion": "Applied Ethics",
    "abstract objects": "Metaphysics",
    "aesthetic experience": "Philosophy of Mind",
    "aesthetic value": "Aesthetics",
    "aim of philosophy": "Philosophical Methodology",
    "analysis of knowledge": "Epistemology",
    "analytic-synthetic distinction": "Philosophy of Language",
    "arguments for theism": "Philosophy of Religion",
    "belief or credence": "Epistemology",
    "capital punishment": "Applied Ethics",
    "causation": "Metaphysics",
    "chinese room": "Philosophy of Mind",
    "concepts": "Philosophy of Mind",
    "consciousness": "Philosophy of Mind",
    "continuum hypothesis": "Logic & Formal Philosophy",
    "cosmological fine-tuning": "Philosophy of Religion",
    "eating animals and animal products": "Applied Ethics",
    "environmental ethics": "Applied Ethics",
    "epistemic justification": "Epistemology",
    "experience machine": "Applied Ethics",
    "extended mind": "Philosophy of Mind",
    "external world": "Metaphysics",
    "footbridge": "Applied Ethics",
    "foundations of mathematics": "Logic & Formal Philosophy",
    "free will": "Metaphysics",
    "gender": "Political & Social Philosophy",
    "gender categories": "Political & Social Philosophy",
    "god": "Philosophy of Religion",
    "grounds of intentionality": "Philosophy of Mind",
    "hard problem of consciousness": "Philosophy of Mind",
    "human genetic engineering": "Applied Ethics",
    "hume": "History of Philosophy",
    "immortality": "Philosophy of Religion",
    "interlevel metaphysics": "Metaphysics",
    "justification": "Epistemology",
    "kant": "History of Philosophy",
    "knowledge": "Epistemology",
    "knowledge claims": "Epistemology",
    "law": "Political & Social Philosophy",
    "laws of nature": "Metaphysics",
    "logic": "Logic & Formal Philosophy",
    "material composition": "Metaphysics",
    "meaning of life": "Ethics & Moral Philosophy",
    "mental content": "Philosophy of Mind",
    "meta-ethics": "Ethics & Moral Philosophy",
    "metaontology": "Metaphysics",
    "metaphilosophy": "Philosophical Methodology",
    "method in history of philosophy": "Philosophical Methodology",
    "method in political philosophy": "Philosophical Methodology",
    "mind": "Philosophy of Mind",
    "mind uploading": "Philosophy of Mind",
    "moral judgment": "Ethics & Moral Philosophy",
    "moral motivation": "Ethics & Moral Philosophy",
    "moral principles": "Ethics & Moral Philosophy",
    "morality": "Ethics & Moral Philosophy",
    "newcomb's problem": "Decision Theory",
    "normative concepts": "Ethics & Moral Philosophy",
    "normative ethics": "Ethics & Moral Philosophy",
    "other minds": "Philosophy of Mind",
    "ought implies can": "Ethics & Moral Philosophy",
    "perceptual experience": "Philosophy of Mind",
    "personal identity": "Metaphysics",
    "philosophical knowledge": "Epistemology",
    "philosophical methods": "Philosophical Methodology",
    "philosophical progress": "Philosophical Methodology",
    "plato": "History of Philosophy",
    "political philosophy": "Political & Social Philosophy",
    "politics": "Political & Social Philosophy",
    "possible worlds": "Metaphysics",
    "practical reason": "Decision Theory",
    "principle of sufficient reason": "Logic & Formal Philosophy",
    "proper names": "Philosophy of Language",
    "properties": "Metaphysics",
    "propositional attitudes": "Philosophy of Mind",
    "propositions": "Philosophy of Language",
    "quantum mechanics": "Philosophy of Science",
    "race": "Political & Social Philosophy",
    "race categories": "Political & Social Philosophy",
    "rational disagreement": "Epistemology",
    "response to external-world skepticism": "Epistemology",
    "science": "Philosophy of Science",
    "semantic content": "Philosophy of Language",
    "sleeping beauty": "Decision Theory",
    "spacetime": "Metaphysics",
    "statue and lump": "Metaphysics",
    "teletransporter": "Metaphysics",
    "temporal ontology": "Metaphysics",
    "theory of reference": "Philosophy of Language",
    "time": "Metaphysics",
    "time travel": "Metaphysics",
    "trolley problem": "Applied Ethics",
    "true contradictions": "Logic & Formal Philosophy",
    "truth": "Philosophy of Language",
    "units of selection": "Philosophy of Science",
    "vagueness": "Philosophy of Language",
    "values in science": "Philosophy of Science",
    "well-being": "Ethics & Moral Philosophy",
    "wittgenstein": "History of Philosophy",
    "zombies": "Philosophy of Mind",
}


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────


def load_all_data():
    """Load response data from final_normalized_100q and demographic data."""
    responses = {}
    demographics = {}

    for model, fname in MODEL_FILES.items():
        with open(NORMALIZED_DIR / fname) as f:
            responses[model] = json.load(f)

    for model, fname in DEMOGRAPHIC_FILES.items():
        with open(DATA_DIR / fname) as f:
            demographics[model] = {p["name"]: p for p in json.load(f)}

    return responses, demographics


def build_matrices(responses):
    """Build aligned 277×100 matrices for all models."""
    # Get common philosopher ordering and question ordering
    # Use human philosopher ordering as reference
    human_data = responses["Human"]
    phil_names = [p["name"] for p in human_data]

    # Get all question keys (should be 100 for each)
    all_keys = set()
    for p in human_data:
        all_keys.update(p["responses"].keys())
    question_keys = sorted(all_keys)

    matrices = {}
    for model, data in responses.items():
        name_to_phil = {p["name"]: p for p in data}
        n_phil = len(phil_names)
        n_q = len(question_keys)
        mat = np.full((n_phil, n_q), np.nan)
        for i, name in enumerate(phil_names):
            if name in name_to_phil:
                resp = name_to_phil[name].get("responses", {})
                for j, qk in enumerate(question_keys):
                    if qk in resp and resp[qk] is not None:
                        mat[i, j] = resp[qk]
        matrices[model] = mat

    return matrices, phil_names, question_keys


def get_question_stem(qkey):
    """Extract question stem from a response key like 'abortion: permissible'."""
    return qkey.split(":")[0].strip().lower()


# ──────────────────────────────────────────────────────────────────────
# 1. Demographic-Position Correlations (Tables 2, 3)
# ──────────────────────────────────────────────────────────────────────


def build_demographic_features(phil_names, demographics_dict):
    """Build binary demographic feature matrix for one model's philosophers."""
    # Collect AOS, AOI, PhD country, PhD year bins
    features = {}  # feature_name -> array of 0/1

    for i, name in enumerate(phil_names):
        demo = demographics_dict.get(name, {})
        # AOS
        for aos in demo.get("areas_of_specialization", []):
            key = f"AOS: {aos}"
            if key not in features:
                features[key] = np.zeros(len(phil_names))
            features[key][i] = 1

        # AOI
        for aoi in demo.get("areas_of_interest", []):
            key = f"AOI: {aoi}"
            if key not in features:
                features[key] = np.zeros(len(phil_names))
            features[key][i] = 1

        # PhD country
        country = demo.get("phd_country", "Unknown")
        if country:
            key = f"PhD: {country}"
            if key not in features:
                features[key] = np.zeros(len(phil_names))
            features[key][i] = 1

        # PhD year bin
        year = demo.get("year_of_phd_degree")
        if year and isinstance(year, (int, float)):
            bin_start = int(year // 5) * 5
            key = f"Year: {bin_start}-{bin_start+4}"
            if key not in features:
                features[key] = np.zeros(len(phil_names))
            features[key][i] = 1

    return features


def compute_demographic_correlations(matrix, question_keys, features):
    """Compute correlations between demographic features and question responses."""
    results = []
    n_features = len(features)
    n_questions = len(question_keys)

    for feat_name, feat_vec in features.items():
        for j, qk in enumerate(question_keys):
            col = matrix[:, j]
            mask = ~np.isnan(col)
            if mask.sum() > 5 and feat_vec[mask].std() > 0 and col[mask].std() > 0:
                r, p = pearsonr(feat_vec[mask], col[mask])
                results.append(
                    {
                        "feature": feat_name,
                        "question": qk,
                        "r": r,
                        "p": p,
                        "abs_r": abs(r),
                    }
                )

    return results


def run_demographic_analysis(matrices, phil_names, question_keys, demographics):
    """Run full demographic-position correlation analysis."""
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC-POSITION CORRELATIONS (Tables 2, 3)")
    print("=" * 80)

    n_tests_total = 0
    bonferroni_alpha = 0.05

    for model in matrices:
        demo_dict = demographics.get(model, {})
        features = build_demographic_features(phil_names, demo_dict)
        corrs = compute_demographic_correlations(
            matrices[model], question_keys, features
        )

        if not corrs:
            print(f"  {model}: No correlations computed (missing demographic data?)")
            continue

        n_tests = len(corrs)
        if model == "Human":
            n_tests_total = n_tests

        abs_rs = [c["abs_r"] for c in corrs]
        max_r = max(abs_rs)
        mean_r = np.mean(abs_rs)

        # Uncorrected significance
        n_sig_uncorrected = sum(1 for c in corrs if c["p"] < 0.05)
        pct_sig_uncorrected = n_sig_uncorrected / n_tests * 100

        # Bonferroni corrected
        bonf_threshold = bonferroni_alpha / n_tests
        n_sig_bonferroni = sum(1 for c in corrs if c["p"] < bonf_threshold)
        pct_sig_bonferroni = n_sig_bonferroni / n_tests * 100

        # Top correlation
        top = max(corrs, key=lambda c: c["abs_r"])

        print(f"\n  {model}:")
        print(f"    N tests:              {n_tests}")
        print(
            f"    Max |r|:              {max_r:.3f} ({top['feature']} -> {top['question']})"
        )
        print(f"    Mean |r|:             {mean_r:.3f}")
        print(f"    % sig (uncorrected):  {pct_sig_uncorrected:.1f}%")
        print(f"    % sig (Bonferroni):   {pct_sig_bonferroni:.2f}%")

        # Top 6 most predictable questions
        corrs_sorted = sorted(corrs, key=lambda c: c["abs_r"], reverse=True)
        print(f"    Top-6 predictable:")
        for c in corrs_sorted[:6]:
            print(f"      |r|={c['abs_r']:.3f}  {c['feature']} -> {c['question']}")

        # Look for specific specialist effects for Table 3
        if model != "Human":
            # Phil Biology -> personal identity
            bio_pi = [
                c
                for c in corrs
                if "Biology" in c["feature"] and "personal identity" in c["question"]
            ]
            if bio_pi:
                for c in bio_pi:
                    print(
                        f"    [Table 3] {c['feature']} -> {c['question']}: r={c['r']:.3f}, p={c['p']:.4f}"
                    )

            # Ancient Phil -> practical reason
            ancient_pr = [
                c
                for c in corrs
                if "Ancient" in c["feature"] and "practical reason" in c["question"]
            ]
            if ancient_pr:
                for c in ancient_pr:
                    print(
                        f"    [Table 3] {c['feature']} -> {c['question']}: r={c['r']:.3f}, p={c['p']:.4f}"
                    )


# ──────────────────────────────────────────────────────────────────────
# 2. Question-Level Predictability (Tables 4, 5, domain_per_model)
# ──────────────────────────────────────────────────────────────────────


def compute_rmse_per_question(human_matrix, model_matrix):
    """Compute RMSE per question between human and model."""
    n_q = human_matrix.shape[1]
    rmse = np.full(n_q, np.nan)
    for j in range(n_q):
        h = human_matrix[:, j]
        m = model_matrix[:, j]
        mask = ~(np.isnan(h) | np.isnan(m))
        if mask.sum() > 0:
            rmse[j] = np.sqrt(np.mean((h[mask] - m[mask]) ** 2))
    return rmse


def run_predictability_analysis(matrices, question_keys):
    """Run question-level and domain-level predictability analysis."""
    print("\n" + "=" * 80)
    print("QUESTION-LEVEL PREDICTABILITY (Tables 4, 5, domain_per_model)")
    print("=" * 80)

    human_mat = matrices["Human"]
    model_names = [m for m in matrices if m != "Human"]

    # Per-model per-question RMSE
    per_model_rmse = {}
    for model in model_names:
        per_model_rmse[model] = compute_rmse_per_question(human_mat, matrices[model])

    # Average RMSE across models per question
    all_rmse = np.array([per_model_rmse[m] for m in model_names])
    avg_rmse = np.nanmean(all_rmse, axis=0)

    # Question stems for domain lookup
    q_stems = [get_question_stem(qk) for qk in question_keys]

    # Table 4: Most/Least predictable
    q_rmse_pairs = [
        (question_keys[j], q_stems[j], avg_rmse[j])
        for j in range(len(question_keys))
        if not np.isnan(avg_rmse[j])
    ]
    q_rmse_pairs.sort(key=lambda x: x[2])

    print("\n  Most predictable questions (Table 4a):")
    for qk, stem, rmse in q_rmse_pairs[:5]:
        domain = DOMAIN_MAP.get(stem, "?")
        print(f"    RMSE={rmse:.3f}  {qk:40s}  ({domain})")

    print("\n  Least predictable questions (Table 4b):")
    for qk, stem, rmse in q_rmse_pairs[-5:][::-1]:
        domain = DOMAIN_MAP.get(stem, "?")
        print(f"    RMSE={rmse:.3f}  {qk:40s}  ({domain})")

    print(
        f"\n  RMSE range: {q_rmse_pairs[0][2]:.3f} to {q_rmse_pairs[-1][2]:.3f} ({q_rmse_pairs[-1][2]/q_rmse_pairs[0][2]:.1f}x)"
    )

    # RMSE vs human ground-truth variance correlation
    from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr

    human_var = np.nanvar(human_mat, axis=0)
    valid_idx = [
        j
        for j in range(len(question_keys))
        if not np.isnan(avg_rmse[j]) and not np.isnan(human_var[j])
    ]
    rmse_vals = [avg_rmse[j] for j in valid_idx]
    var_vals = [human_var[j] for j in valid_idx]
    r_pear, p_pear = _pearsonr(rmse_vals, var_vals)
    r_spear, p_spear = _spearmanr(rmse_vals, var_vals)
    print(f"\n  RMSE vs human variance correlation:")
    print(f"    Pearson  r={r_pear:.3f} (p={p_pear:.2e})")
    print(f"    Spearman ρ={r_spear:.3f} (p={p_spear:.2e})")

    # Table 5: Domain-level RMSE + human variance
    human_var = np.nanvar(human_mat, axis=0)
    domain_rmses = defaultdict(list)
    domain_hvars = defaultdict(list)
    for j, stem in enumerate(q_stems):
        domain = DOMAIN_MAP.get(stem)
        if domain and not np.isnan(avg_rmse[j]):
            domain_rmses[domain].append(avg_rmse[j])
        if domain and not np.isnan(human_var[j]):
            domain_hvars[domain].append(human_var[j])

    domain_avg = [
        (d, np.mean(rs), np.mean(domain_hvars.get(d, [float("nan")])), len(rs))
        for d, rs in domain_rmses.items()
    ]
    domain_avg.sort(key=lambda x: x[1])

    print("\n  Domain predictability (Table 5):")
    print(f"    {'Rank':<5} {'Domain':<35} {'RMSE':>6} {'H.Var':>7} {'N':>4}")
    print("    " + "-" * 63)
    for rank, (domain, rmse, hvar, n) in enumerate(domain_avg, 1):
        hvar_str = f"{hvar:.3f}" if not np.isnan(hvar) else "  ---"
        print(f"    {rank:<5} {domain:<35} {rmse:.3f}  {hvar_str}  {n:>3}")

    print(
        f"\n  Domain RMSE range: {domain_avg[0][1]:.3f} to {domain_avg[-1][1]:.3f} ({domain_avg[-1][1]/domain_avg[0][1]:.1f}x)"
    )

    # Domain-level RMSE vs human variance correlation
    from scipy.stats import pearsonr as _pearsonr2, spearmanr as _spearmanr2

    dom_rmse_vals = [rmse for _, rmse, hvar, _ in domain_avg if not np.isnan(hvar)]
    dom_hvar_vals = [hvar for _, rmse, hvar, _ in domain_avg if not np.isnan(hvar)]
    if len(dom_rmse_vals) >= 3:
        r_d, p_d = _pearsonr2(dom_rmse_vals, dom_hvar_vals)
        s_d, ps_d = _spearmanr2(dom_rmse_vals, dom_hvar_vals)
        print(
            f"  Domain RMSE vs human variance: Pearson r={r_d:.3f} (p={p_d:.3f}), Spearman ρ={s_d:.3f} (p={ps_d:.3f}), n={len(dom_rmse_vals)} domains"
        )

    # Domain per model table
    print("\n  Per-model RMSE by domain (tab:domain_per_model):")
    # Select domains to show
    domains_to_show = [
        "Philosophy of Science",
        "Philosophy of Mind",
        "Metaphysics",
        "Applied Ethics",
        "Logic & Formal Philosophy",
        "Philosophy of Religion",
    ]

    header = f"    {'Domain':<30}"
    for m in model_names:
        short = (
            m.replace("Claude ", "")
            .replace("Llama 3.1 ", "Llama ")
            .replace(" (FT)", "-FT")
        )
        header += f" {short:>10}"
    print(header)
    print("    " + "-" * (30 + 11 * len(model_names)))

    for domain in domains_to_show:
        row = f"    {domain:<30}"
        for model in model_names:
            # Get per-question RMSE for this model+domain
            domain_q_idx = [
                j for j, s in enumerate(q_stems) if DOMAIN_MAP.get(s) == domain
            ]
            model_rmse_vals = [
                per_model_rmse[model][j]
                for j in domain_q_idx
                if not np.isnan(per_model_rmse[model][j])
            ]
            if model_rmse_vals:
                row += f" {np.mean(model_rmse_vals):>10.2f}"
            else:
                row += f" {'---':>10}"
        print(row)


# ──────────────────────────────────────────────────────────────────────
# 3. PCA Analysis (Table 6, PC1 comparison)
# ──────────────────────────────────────────────────────────────────────


def impute_pca_iterative(X, ncp=5, max_iter=1000, tol=1e-6):
    """Iterative PCA imputation (equivalent to R's missMDA package).

    Following Bourget & Chalmers (2023): initialize missing values with column
    means, fit PCA with ncp components, reconstruct and replace only missing
    entries, iterate until convergence.
    """
    from sklearn.decomposition import PCA as SkPCA

    X_imp = X.copy()
    missing = np.isnan(X)
    col_means = np.nanmean(X, axis=0)
    global_mean = np.nanmean(X)
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    for j in range(X.shape[1]):
        X_imp[missing[:, j], j] = col_means[j]
    for it in range(max_iter):
        X_old = X_imp.copy()
        means = X_imp.mean(axis=0)
        X_c = X_imp - means
        pca = SkPCA(n_components=min(ncp, min(X.shape) - 1))
        scores = pca.fit_transform(X_c)
        X_rec = scores @ pca.components_ + means
        X_imp[missing] = X_rec[missing]
        diff = np.sum((X_imp[missing] - X_old[missing]) ** 2)
        total_sq = np.sum(X_imp[missing] ** 2) + 1e-10
        if diff / total_sq < tol:
            break
    return X_imp


def safe_corr_matrix(X):
    """Correlation matrix that handles zero-variance columns."""
    C = np.corrcoef(X.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def pairwise_corr_matrix(X):
    """Pairwise-deletion correlation matrix (no imputation).

    For each pair of columns, uses only rows where both are non-NaN.
    Used for question correlation structure analysis (Table 7/8)
    where imputation would introduce artificial structure.
    """
    n = X.shape[1]
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                r, _ = pearsonr(X[mask, i], X[mask, j])
                if np.isnan(r):
                    r = 0.0
                C[i, j] = r
                C[j, i] = r
    return C


def run_pca_analysis(matrices, question_keys):
    """Run full PCA analysis following B&C methodology."""
    print("\n" + "=" * 80)
    print("PCA ANALYSIS (Table 6, PC1 comparison)")
    print("=" * 80)

    # Since data is already 100 questions (one per stem, positive option selected),
    # we use all 100 columns directly. No further variable selection needed.

    from sklearn.decomposition import PCA as SkPCA

    n_components = 6
    pca_results = {}

    for model, mat in matrices.items():
        # Impute missing values via iterative PCA (missMDA equivalent)
        mat_imputed = impute_pca_iterative(mat, ncp=5)
        corr_mat = safe_corr_matrix(mat_imputed)

        # Run PCA on imputed data (centered, not scaled — matches B&C)
        pca = SkPCA(n_components=n_components)
        centered = mat_imputed - mat_imputed.mean(axis=0)
        pca.fit(centered)
        var_exp = pca.explained_variance_ratio_[:n_components]
        # Loadings = components transposed (each column = one PC's loadings)
        evecs = pca.components_.T

        n_sig = int(np.sum(var_exp >= 0.02))
        var6 = float(np.sum(var_exp))

        # Top loadings for each component
        top_loadings = []
        for c in range(n_components):
            loadings_c = evecs[:, c]
            top_idx = np.argsort(np.abs(loadings_c))[::-1][:5]
            top = [(question_keys[i], float(loadings_c[i])) for i in top_idx]
            top_loadings.append(top)

        # Pairwise-deletion correlation matrix (no imputation) for Table 7/8
        corr_mat_pairwise = pairwise_corr_matrix(mat)

        pca_results[model] = {
            "var_explained": var_exp,
            "var6": var6,
            "n_significant": n_sig,
            "loadings": evecs,
            "top_loadings": top_loadings,
            "corr_matrix": corr_mat,  # imputed (for PCA / Table 6)
            "corr_matrix_pairwise": corr_mat_pairwise,  # raw (for Table 7/8)
        }

        print(f"\n  {model}:")
        print(f"    Sig components (>=2%): {n_sig}")
        print(f"    Var(6): {var6*100:.1f}%")
        for c in range(min(n_sig, n_components)):
            print(f"    PC{c+1} ({var_exp[c]*100:.1f}%): ", end="")
            for q, l in top_loadings[c][:3]:
                print(f"{l:+.2f} {q.split(':')[0]}", end="; ")
            print()

    # Comparison: each model vs Human
    print("\n  --- Model vs Human comparison (Table 6) ---")
    human_loadings = pca_results["Human"]["loadings"]
    human_corr = pca_results["Human"]["corr_matrix"]

    print(f"\n  {'Model':<22} {'Var(6)':>8} {'Elem r':>8} {'Load r':>8} {'Overlap':>8}")
    print("  " + "-" * 58)
    print(
        f"  {'Human':<22} {pca_results['Human']['var6']*100:>7.1f}% {'---':>8} {'---':>8} {'---':>8}"
    )

    for model in matrices:
        if model == "Human":
            continue

        model_loadings = pca_results[model]["loadings"]
        model_corr = pca_results[model]["corr_matrix"]

        # Element-wise correlation of question correlation matrices
        # Upper triangle only
        idx_upper = np.triu_indices(human_corr.shape[0], k=1)
        h_upper = human_corr[idx_upper]
        m_upper = model_corr[idx_upper]
        elem_r, _ = pearsonr(h_upper, m_upper)

        # Loading correlation (flattened)
        h_load_flat = human_loadings.flatten()
        m_load_flat = model_loadings.flatten()
        load_r, _ = pearsonr(h_load_flat, m_load_flat)

        # Top-5 question overlap per component (with alignment)
        n_comp = n_components
        used = set()
        alignment = []
        for i in range(n_comp):
            best_j, best_corr = -1, -1
            for j in range(n_comp):
                if j not in used:
                    r = abs(
                        np.corrcoef(human_loadings[:, i], model_loadings[:, j])[0, 1]
                    )
                    if not np.isnan(r) and r > best_corr:
                        best_corr = r
                        best_j = j
            if best_j >= 0:
                used.add(best_j)
            alignment.append(best_j if best_j >= 0 else i)

        overlap_counts = []
        for i in range(n_comp):
            h_top = set(q for q, _ in pca_results["Human"]["top_loadings"][i])
            m_idx = alignment[i]
            m_top = set(q for q, _ in pca_results[model]["top_loadings"][m_idx])
            overlap_counts.append(len(h_top & m_top))
        mean_overlap = np.mean(overlap_counts)

        print(
            f"  {model:<22} {pca_results[model]['var6']*100:>7.1f}% {elem_r:>8.3f} {load_r:>8.3f} {mean_overlap:>6.1f}/5"
        )

    # PC1 comparison table
    print("\n  --- PC1 Top-5 Loadings (tab:pc1_comparison) ---")
    for model in ["Human", "GPT-4o", "Claude Sonnet 4.5", "Llama 3.1 8B"]:
        print(f"\n  {model}:")
        for q, l in pca_results[model]["top_loadings"][0]:
            print(f"    {l:+.3f}  {q}")

    # Per-component loading correlations (appendix)
    print(
        "\n  --- Per-component loading correlations (appendix tab:model_pc_comparison) ---"
    )
    header = f"  {'Model':<22}"
    for c in range(n_components):
        header += f" {'PC'+str(c+1):>6}"
    print(header)
    print("  " + "-" * (22 + 7 * n_components))

    for model in matrices:
        if model == "Human":
            continue
        row = f"  {model:<22}"
        for c in range(n_components):
            r = abs(
                np.corrcoef(human_loadings[:, c], pca_results[model]["loadings"][:, c])[
                    0, 1
                ]
            )
            row += f" {r:>6.2f}"
        print(row)

    return pca_results


# ──────────────────────────────────────────────────────────────────────
# 4. Question Correlation Structure (Table 7) + Fine-tuning (Table 8)
# ──────────────────────────────────────────────────────────────────────


def mantel_test(corr1, corr2, n_perms=9999):
    """Mantel test: correlation between distance/correlation matrices."""
    idx = np.triu_indices(corr1.shape[0], k=1)
    x = corr1[idx]
    y = corr2[idx]
    observed_r, _ = pearsonr(x, y)

    count = 0
    n = corr1.shape[0]
    for _ in range(n_perms):
        perm = np.random.permutation(n)
        perm_corr = corr2[np.ix_(perm, perm)]
        perm_y = perm_corr[idx]
        perm_r, _ = pearsonr(x, perm_y)
        if abs(perm_r) >= abs(observed_r):
            count += 1

    p_value = (count + 1) / (n_perms + 1)
    return observed_r, p_value


def rv_coefficient(corr1, corr2):
    """RV coefficient: multivariate generalization of R²."""
    # Center the matrices
    n = corr1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    A = H @ corr1 @ H
    B = H @ corr2 @ H

    numerator = np.trace(A @ B)
    denominator = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def kl_divergence_distributions(p_vals, q_vals, n_bins=20):
    """KL divergence between two distributions of correlation values."""
    bins = np.linspace(-1, 1, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)

    # Add epsilon
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def js_divergence_distributions(p_vals, q_vals, n_bins=20):
    """JS divergence between two distributions of correlation values."""
    bins = np.linspace(-1, 1, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)

    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    m = 0.5 * (p_hist + q_hist)
    return float(
        0.5 * np.sum(p_hist * np.log(p_hist / m))
        + 0.5 * np.sum(q_hist * np.log(q_hist / m))
    )


def compute_response_entropy(matrix):
    """Average per-question Shannon entropy."""
    entropies = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            # Bin into the 5 response categories
            bins = [0, 0.125, 0.375, 0.625, 0.875, 1.001]
            counts, _ = np.histogram(valid, bins=bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
    return np.mean(entropies) if entropies else 0.0


def compute_response_kl(human_matrix, model_matrix):
    """KL divergence of response distributions (flattened, binned)."""
    h_valid = human_matrix[~np.isnan(human_matrix)]
    m_valid = model_matrix[~np.isnan(model_matrix)]

    bins = np.linspace(0, 1, 21)
    h_hist, _ = np.histogram(h_valid, bins=bins, density=True)
    m_hist, _ = np.histogram(m_valid, bins=bins, density=True)

    eps = 1e-10
    h_hist = h_hist + eps
    m_hist = m_hist + eps
    h_hist = h_hist / h_hist.sum()
    m_hist = m_hist / m_hist.sum()

    return float(np.sum(h_hist * np.log(h_hist / m_hist)))


def run_question_correlation_analysis(matrices, question_keys, pca_results):
    """Run question correlation structure analysis (Table 7) and fine-tuning (Table 8).

    Uses pairwise-deletion correlation matrices (NOT imputed) to assess
    whether LLMs preserve the observed inter-question correlation structure.
    Imputation is reserved for PCA only (Table 6).
    """
    print("\n" + "=" * 80)
    print("QUESTION CORRELATION STRUCTURE (Table 7) — pairwise deletion")
    print("=" * 80)

    human_corr = pca_results["Human"]["corr_matrix_pairwise"]
    idx_upper = np.triu_indices(human_corr.shape[0], k=1)
    human_upper = human_corr[idx_upper]

    # Specific correlations mentioned in text
    q_stems = [get_question_stem(qk) for qk in question_keys]

    # Find physicalism -> atheism and physicalism -> naturalism correlations
    mind_idx = [j for j, s in enumerate(q_stems) if "mind" == s]
    god_idx = [j for j, s in enumerate(q_stems) if "god" == s]

    if mind_idx and god_idx:
        mi, gi = mind_idx[0], god_idx[0]
        r_mind_god = human_corr[mi, gi]
        print(f"\n  Inline: mind <-> god correlation: r = {r_mind_god:.3f}")

    print(
        f"\n  {'Model':<22} {'Elem r':>8} {'Mantel p':>10} {'RV':>8} {'KL':>8} {'JS':>8}"
    )
    print("  " + "-" * 62)

    for model in matrices:
        if model == "Human":
            continue

        model_corr = pca_results[model]["corr_matrix_pairwise"]
        model_upper = model_corr[idx_upper]

        # Element-wise correlation (already computed, but let's redo cleanly with Mantel)
        elem_r, mantel_p = mantel_test(human_corr, model_corr, n_perms=999)

        # RV coefficient
        rv = rv_coefficient(human_corr, model_corr)

        # KL and JS on correlation distributions
        kl = kl_divergence_distributions(human_upper, model_upper)
        js = js_divergence_distributions(human_upper, model_upper)

        sig = (
            "***"
            if mantel_p < 0.001
            else ("**" if mantel_p < 0.01 else ("*" if mantel_p < 0.05 else ""))
        )

        print(
            f"  {model:<22} {elem_r:>7.3f}{sig:<3} {mantel_p:>10.4f} {rv:>8.3f} {kl:>8.3f} {js:>8.3f}"
        )

    # Table 8: Fine-tuning trade-off
    print("\n" + "=" * 80)
    print("FINE-TUNING TRADE-OFF (Table 8)")
    print("=" * 80)

    base_mat = matrices["Llama 3.1 8B"]
    ft_mat = matrices["Llama 3.1 8B (FT)"]
    human_mat = matrices["Human"]

    base_entropy = compute_response_entropy(base_mat)
    ft_entropy = compute_response_entropy(ft_mat)

    base_resp_kl = compute_response_kl(human_mat, base_mat)
    ft_resp_kl = compute_response_kl(human_mat, ft_mat)

    base_corr = pca_results["Llama 3.1 8B"]["corr_matrix_pairwise"]
    ft_corr = pca_results["Llama 3.1 8B (FT)"]["corr_matrix_pairwise"]
    human_corr_upper = human_corr[idx_upper]
    base_upper = base_corr[idx_upper]
    ft_upper = ft_corr[idx_upper]

    base_corr_kl = kl_divergence_distributions(human_corr_upper, base_upper)
    ft_corr_kl = kl_divergence_distributions(human_corr_upper, ft_upper)
    base_corr_js = js_divergence_distributions(human_corr_upper, base_upper)
    ft_corr_js = js_divergence_distributions(human_corr_upper, ft_upper)

    print(
        f"\n  {'':>25} {'Entropy':>10} {'Resp KL':>10} {'Corr KL':>10} {'Corr JS':>10}"
    )
    print("  " + "-" * 68)
    print(
        f"  {'Llama 3.1 8B (Base)':<25} {base_entropy:>10.3f} {base_resp_kl:>10.2f} {base_corr_kl:>10.3f} {base_corr_js:>10.3f}"
    )
    print(
        f"  {'Llama 3.1 8B (FT)':<25} {ft_entropy:>10.3f} {ft_resp_kl:>10.2f} {ft_corr_kl:>10.3f} {ft_corr_js:>10.3f}"
    )

    # Count uniform-response questions
    for label, mat in [("Base", base_mat), ("FT", ft_mat)]:
        n_uniform = 0
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 1 and np.var(valid) < 0.001:
                n_uniform += 1
        print(f"  {label} uniform-response questions (var < 0.001): {n_uniform}")

    # Per-question entropy change (appendix)
    print("\n  --- Fine-tuning per-question entropy changes (appendix) ---")
    q_entropy_changes = []
    for j in range(base_mat.shape[1]):
        base_col = base_mat[:, j][~np.isnan(base_mat[:, j])]
        ft_col = ft_mat[:, j][~np.isnan(ft_mat[:, j])]
        if len(base_col) > 1 and len(ft_col) > 1:
            bins = [0, 0.125, 0.375, 0.625, 0.875, 1.001]
            base_counts, _ = np.histogram(base_col, bins=bins)
            ft_counts, _ = np.histogram(ft_col, bins=bins)

            base_p = base_counts / base_counts.sum()
            ft_p = ft_counts / ft_counts.sum()

            base_h = -sum(p * np.log2(p) for p in base_p if p > 0)
            ft_h = -sum(p * np.log2(p) for p in ft_p if p > 0)

            q_entropy_changes.append((question_keys[j], base_h, ft_h, ft_h - base_h))

    q_entropy_changes.sort(key=lambda x: x[3], reverse=True)
    print("\n  Top 5 increased diversity:")
    for qk, bh, fh, dh in q_entropy_changes[:5]:
        pct = (dh / bh * 100) if bh > 0 else float("inf")
        print(f"    {qk:40s} Base={bh:.2f} FT={fh:.2f} ΔH={dh:+.2f} ({pct:+.0f}%)")

    print("\n  Top 5 decreased diversity:")
    for qk, bh, fh, dh in q_entropy_changes[-5:][::-1]:
        pct = (dh / bh * 100) if bh > 0 else 0
        print(f"    {qk:40s} Base={bh:.2f} FT={fh:.2f} ΔH={dh:+.2f} ({pct:+.0f}%)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("RECOMPUTING ALL PAPER TABLES FROM final_normalized_100q/")
    print("Data: 277 philosophers × 100 questions (positive-option, B&C methodology)")
    print("=" * 80)

    responses, demographics = load_all_data()
    matrices, phil_names, question_keys = build_matrices(responses)

    print(f"\nLoaded {len(matrices)} datasets")
    print(f"Philosopher count: {len(phil_names)}")
    print(f"Question count: {len(question_keys)}")
    for model, mat in matrices.items():
        non_nan = np.sum(~np.isnan(mat))
        total = mat.size
        print(f"  {model:25s}: {non_nan}/{total} ({non_nan/total*100:.1f}%)")

    # 1. Demographic correlations
    run_demographic_analysis(matrices, phil_names, question_keys, demographics)

    # 2. Question-level predictability
    run_predictability_analysis(matrices, question_keys)

    # 3. PCA
    pca_results = run_pca_analysis(matrices, question_keys)

    # 4. Question correlation structure + fine-tuning
    run_question_correlation_analysis(matrices, question_keys, pca_results)

    print("\n" + "=" * 80)
    print("ALL COMPUTATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(42)
    main()
