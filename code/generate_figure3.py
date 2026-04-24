#!/usr/bin/env python3
"""
Generate Figure 3: Side-by-side comparison figures for the paper.
  (a) Human vs Claude Sonnet 4.5
  (b) Human vs Llama 3.1 8B

Reads from final_normalized_100q/ (positive-option methodology).
Following Bourget & Chalmers (2023), we select the most popular option
for each non-binary question; our positive-option selection approximates
this by choosing the affirmative/popular side of each contradictory pair.

Data is normalized to [0,1]:
  0.00 = Reject (-2)
  0.25 = Lean Against (-1)
  0.50 = Agnostic (0)
  0.75 = Lean Toward (+1)
  1.00 = Accept (+2)

Color scale: Red (Reject/0) -> White (Agnostic/0.5) -> Blue (Accept/1)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
from pathlib import Path

# Resolve data dir relative to this script: release/code/*.py → release/data/
# Output figures go to release/paper/figures/ by default.
import os

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", _SCRIPT_DIR.parent / "paper" / "figures")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "Human": DATA_DIR / "final_normalized_100q" / "human_survey_normalized.json",
    "Claude Sonnet 4.5": DATA_DIR
    / "final_normalized_100q"
    / "sonnet45_normalized.json",
    "Llama 3.1 8B": DATA_DIR / "final_normalized_100q" / "llama3p18b_normalized.json",
}


def load_dataset(filepath):
    with open(filepath) as f:
        return json.load(f)


def create_response_matrix(data):
    """Create 277 x 100 matrix from normalized JSON data."""
    # Get sorted question keys
    all_keys = set()
    for phil in data:
        if "responses" in phil:
            all_keys.update(phil["responses"].keys())
    question_keys = sorted(all_keys)

    n_phils = len(data)
    n_questions = len(question_keys)
    matrix = np.full((n_phils, n_questions), np.nan)

    for i, phil in enumerate(data):
        if "responses" in phil:
            for j, qkey in enumerate(question_keys):
                val = phil["responses"].get(qkey)
                if val is not None:
                    matrix[i, j] = val

    return matrix, question_keys


# Diverging colormap: Red (0/Reject) -> White (0.5/Agnostic) -> Blue (1/Accept)
COLORS = [
    (0.698, 0.094, 0.168),  # 0.00: Dark red - Reject
    (0.937, 0.541, 0.384),  # 0.25: Light red - Lean Against
    (0.969, 0.969, 0.969),  # 0.50: Near white - Agnostic
    (0.573, 0.773, 0.871),  # 0.75: Light blue - Lean Toward
    (0.129, 0.400, 0.675),  # 1.00: Dark blue - Accept
]
CMAP = LinearSegmentedColormap.from_list("diverging_rwb", COLORS, N=256)
CMAP.set_bad(color="#d9d9d9")  # Gray for NaN/missing


def compute_per_question_variance(matrix):
    """Average within-question variance (excluding NaN)."""
    variances = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            variances.append(np.var(valid))
    return np.mean(variances) if variances else 0.0


def count_zero_variance_questions(matrix, threshold=0.001):
    """Count questions with near-zero variance."""
    count = 0
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1 and np.var(valid) < threshold:
            count += 1
    return count


def create_comparison_figure(human_matrix, llm_matrix, llm_name, output_basename):
    """Create a 1x2 side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    # Human panel
    ax1 = axes[0]
    human_masked = np.ma.masked_invalid(human_matrix)
    ax1.imshow(
        human_masked, aspect="auto", cmap=CMAP, norm=norm, interpolation="nearest"
    )
    ax1.set_title("Human Philosophers", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Questions (100)", fontsize=11)
    ax1.set_ylabel("Philosophers (277)", fontsize=11)
    ax1.set_xticks([0, 25, 50, 75, 99])
    ax1.set_xticklabels(["1", "25", "50", "75", "100"])
    ax1.set_yticks([0, 69, 138, 207, 276])
    ax1.set_yticklabels(["1", "70", "139", "208", "277"])

    human_var = compute_per_question_variance(human_matrix)
    human_missing = np.isnan(human_matrix).sum() / human_matrix.size * 100
    ax1.text(
        0.02,
        0.98,
        f"Missing: {human_missing:.0f}%\nPer-Q Var: {human_var:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # LLM panel
    ax2 = axes[1]
    llm_masked = np.ma.masked_invalid(llm_matrix)
    im = ax2.imshow(
        llm_masked, aspect="auto", cmap=CMAP, norm=norm, interpolation="nearest"
    )
    ax2.set_title(f"LLM Simulations ({llm_name})", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Questions (100)", fontsize=11)
    ax2.set_ylabel("Philosophers (277)", fontsize=11)
    ax2.set_xticks([0, 25, 50, 75, 99])
    ax2.set_xticklabels(["1", "25", "50", "75", "100"])
    ax2.set_yticks([0, 69, 138, 207, 276])
    ax2.set_yticklabels(["1", "70", "139", "208", "277"])

    llm_var = compute_per_question_variance(llm_matrix)
    llm_missing = np.isnan(llm_matrix).sum() / llm_matrix.size * 100
    ax2.text(
        0.02,
        0.98,
        f"Missing: {llm_missing:.0f}%\nPer-Q Var: {llm_var:.3f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(
        [
            "Reject\n(-2)",
            "Lean Against\n(-1)",
            "Agnostic\n(0)",
            "Lean Toward\n(+1)",
            "Accept\n(+2)",
        ]
    )
    cbar.ax.tick_params(labelsize=9)

    # Missing data legend
    gray_patch = mpatches.Patch(color="#d9d9d9", label="Missing data")
    fig.legend(
        handles=[gray_patch], loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98)
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.savefig(
        f"{output_basename}.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(f"{output_basename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()

    return human_var, llm_var


def main():
    print("=" * 80)
    print("GENERATING FIGURE 3: Human vs LLM Comparisons")
    print("Methodology: positive-option selection following Bourget & Chalmers (2023)")
    print("  B&C select the most popular option for non-binary questions;")
    print("  we select the affirmative/positive option for contradictory pairs.")
    print("=" * 80)
    print()

    # Load datasets
    print("Loading datasets...")
    human_data = load_dataset(DATASET_FILES["Human"])
    sonnet_data = load_dataset(DATASET_FILES["Claude Sonnet 4.5"])
    llama_data = load_dataset(DATASET_FILES["Llama 3.1 8B"])

    # Create matrices
    print("Creating response matrices...")
    human_matrix, q_keys = create_response_matrix(human_data)
    sonnet_matrix, _ = create_response_matrix(sonnet_data)
    llama_matrix, _ = create_response_matrix(llama_data)

    print(
        f"  Human:  {human_matrix.shape}, Missing={np.isnan(human_matrix).sum()/human_matrix.size*100:.1f}%"
    )
    print(
        f"  Sonnet: {sonnet_matrix.shape}, Missing={np.isnan(sonnet_matrix).sum()/sonnet_matrix.size*100:.1f}%"
    )
    print(
        f"  Llama:  {llama_matrix.shape}, Missing={np.isnan(llama_matrix).sum()/llama_matrix.size*100:.1f}%"
    )
    print()

    # Generate Figure 3a: Human vs Sonnet
    print("Generating Figure 3a: Human vs Claude Sonnet 4.5...")
    h_var, s_var = create_comparison_figure(
        human_matrix,
        sonnet_matrix,
        "Claude Sonnet 4.5",
        str(OUTPUT_DIR / "figure1_human_vs_sonnet"),
    )
    print(
        f"  Human Per-Q Var: {h_var:.3f}, Sonnet Per-Q Var: {s_var:.3f} ({h_var/s_var:.1f}x lower)"
    )
    print(
        f"  Sonnet zero-var questions: {count_zero_variance_questions(sonnet_matrix)}"
    )
    print()

    # Generate Figure 3b: Human vs Llama
    print("Generating Figure 3b: Human vs Llama 3.1 8B...")
    h_var2, l_var = create_comparison_figure(
        human_matrix,
        llama_matrix,
        "Llama 3.1 8B",
        str(OUTPUT_DIR / "figure1_human_vs_llama"),
    )
    print(
        f"  Human Per-Q Var: {h_var2:.3f}, Llama Per-Q Var: {l_var:.3f} ({h_var2/l_var:.1f}x lower)"
    )
    print(f"  Llama zero-var questions: {count_zero_variance_questions(llama_matrix)}")
    print()

    print("=" * 80)
    print("FIGURES SAVED:")
    print("  figure1_human_vs_sonnet.png/pdf")
    print("  figure1_human_vs_llama.png/pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
