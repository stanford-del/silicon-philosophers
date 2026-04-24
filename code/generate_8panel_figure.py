#!/usr/bin/env python3
"""
Generate 8-panel Figure: Matrix visualization showing heterogeneity collapse
Compares Human vs 7 LLMs side-by-side.

Reads from final_normalized_100q/ (positive-option methodology).
Following Bourget & Chalmers (2023), we select the most popular option
for each non-binary question; our positive-option selection approximates
this by choosing the affirmative/positive side of each contradictory pair.

Data is normalized to 0-1:
  0.00 = Reject (-2)
  0.25 = Lean Against (-1)
  0.50 = Agnostic (0)
  0.75 = Lean Toward (+1)
  1.00 = Accept (+2)

Color scale: Red (Reject/0) → White (Agnostic/0.5) → Blue (Accept/1)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
from pathlib import Path

# Resolve data/output dirs relative to this script.
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", _SCRIPT_DIR.parent / "paper" / "figures")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_NORM = DATA_DIR / "final_normalized_100q"
DATASET_FILES = {
    "Human": _NORM / "human_survey_normalized.json",
    "Claude Sonnet 4.5": _NORM / "sonnet45_normalized.json",
    "GPT-4o": _NORM / "openai_gpt4o_normalized.json",
    "GPT-5.1": _NORM / "gpt51_normalized.json",
    "Llama 3.1 8B": _NORM / "llama3p18b_normalized.json",
    "Llama 3.1 8B (FT)": _NORM / "llama3p18b_finetuned_normalized.json",
    "Mistral 7B": _NORM / "mistral7b_normalized.json",
    "Qwen 3 4B": _NORM / "qwen3-4b_normalized.json",
}


def load_dataset(filepath):
    with open(filepath) as f:
        return json.load(f)


def create_response_matrix(data):
    """Create N x 100 matrix from normalized JSON data."""
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

    return matrix


def compute_per_question_variance(matrix):
    """Average within-question variance (excluding NaN)."""
    variances = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            variances.append(np.var(valid))
    return np.mean(variances) if variances else 0.0


def create_8panel_figure():
    """Create 8-panel comparison figure"""
    print("=" * 80)
    print("GENERATING 8-PANEL FIGURE")
    print("Methodology: positive-option selection following Bourget & Chalmers (2023)")
    print("=" * 80)
    print()

    # Load all datasets from normalized JSON
    print("Loading datasets from final_normalized_100q/...")

    model_order = [
        "Human",
        "Claude Sonnet 4.5",
        "GPT-4o",
        "GPT-5.1",
        "Llama 3.1 8B",
        "Llama 3.1 8B (FT)",
        "Mistral 7B",
        "Qwen 3 4B",
    ]

    # Create matrices for all models
    matrices = {}
    print("Creating response matrices...")
    for model_name in model_order:
        data = load_dataset(DATASET_FILES[model_name])
        matrix = create_response_matrix(data)
        matrices[model_name] = matrix

        missing_pct = np.isnan(matrix).sum() / matrix.size * 100
        per_q_var = compute_per_question_variance(matrix)
        print(
            f"  {model_name:25s}: {matrix.shape}, Missing={missing_pct:5.1f}%, Per-Q Var={per_q_var:.4f}"
        )
    print()

    # Create diverging colormap: Red (0/Reject) → White (0.5/Agnostic) → Blue (1/Accept)
    colors = [
        (0.698, 0.094, 0.168),  # 0.00: Dark red - Reject
        (0.937, 0.541, 0.384),  # 0.25: Light red - Lean Against
        (0.969, 0.969, 0.969),  # 0.50: Near white - Agnostic
        (0.573, 0.773, 0.871),  # 0.75: Light blue - Lean Toward
        (0.129, 0.400, 0.675),  # 1.00: Dark blue - Accept
    ]
    cmap = LinearSegmentedColormap.from_list("diverging_rwb", colors, N=256)
    cmap.set_bad(color="#d9d9d9")  # Gray for NaN/missing

    # Create figure with 2 rows × 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Use TwoSlopeNorm to ensure 0.5 (Agnostic) maps to white
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    # Plot each model
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        matrix = matrices[model_name]
        masked_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(
            masked_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
        )

        # Title
        ax.set_title(model_name, fontsize=12, fontweight="bold")

        # Labels only on left and bottom edges
        if idx % 4 == 0:  # Left column
            ax.set_ylabel("Philosophers (277)", fontsize=10)
            ax.set_yticks([0, 69, 138, 207, 276])
            ax.set_yticklabels(["1", "70", "139", "208", "277"])
        else:
            ax.set_yticks([])

        if idx >= 4:  # Bottom row
            ax.set_xlabel("Questions (100)", fontsize=10)
            ax.set_xticks([0, 25, 50, 75, 99])
            ax.set_xticklabels(["1", "25", "50", "75", "100"])
        else:
            ax.set_xticks([])

        # Add per-question variance in corner
        per_q_var = compute_per_question_variance(matrix)

        # Color code: red for human, blue for LLMs
        if model_name == "Human":
            text_color = "#dc2626"
        else:
            text_color = "#2563eb"

        ax.text(
            0.98,
            0.02,
            f"Per-Q Var: {per_q_var:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            color=text_color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor=text_color,
                linewidth=1.5,
            ),
        )

    # Add shared colorbar at bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(
        [
            "Reject\n(−2)",
            "Lean Against\n(−1)",
            "Agnostic\n(0)",
            "Lean Toward\n(+1)",
            "Accept\n(+2)",
        ]
    )
    cbar.ax.tick_params(labelsize=10)

    # Add legend for missing data
    gray_patch = mpatches.Patch(color="#d9d9d9", label="Missing data")
    fig.legend(
        handles=[gray_patch],
        loc="upper right",
        fontsize=10,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Main title
    fig.suptitle(
        "Response Pattern Visualization: Human vs. 7 LLMs",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Subtitle
    fig.text(
        0.5,
        0.94,
        "Each cell shows one philosopher's response to one question (277 philosophers × 100 questions). "
        "Human responses show diverse patterns; LLM responses form uniform vertical bands.",
        ha="center",
        fontsize=11,
        style="italic",
        color="#444444",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    output_png = OUTPUT_DIR / "figure1_8panel_bc.png"
    output_pdf = OUTPUT_DIR / "figure1_8panel_bc.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")

    print("=" * 80)
    print("FIGURE SAVED")
    print("=" * 80)
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")
    print()
    print("Done.")

    plt.close()


if __name__ == "__main__":
    create_8panel_figure()
