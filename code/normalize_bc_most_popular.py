#!/usr/bin/env python3
"""
Normalize to 100 questions using hybrid approach:
  - Binary (contradictory pair) stems: keep positive option with complement
    recovery (if only the negative side was answered, invert: 1.0 - value)
  - Multi-option stems: select the most popular option among human respondents
    following Bourget & Chalmers (2023) methodology

Output: final_normalized_100q/ (overwrites existing files)
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# Binary (contradictory pair) stems: positive -> negative mapping
# For these stems, we keep the positive option and use complement recovery
# (if only the negative side was answered, value = 1.0 - negative_value)
BINARY_PAIRS = {
    "a priori knowledge": ("yes", "no"),
    "abortion": ("permissible", "impermissible"),
    "analytic-synthetic distinction": ("yes", "no"),
    "capital punishment": ("permissible", "impermissible"),
    "environmental ethics": ("anthropocentric", "non-anthropocentric"),
    "epistemic justification": ("externalism", "internalism"),
    "experience machine": ("yes", "no"),
    "extended mind": ("yes", "no"),
    "footbridge": ("push", "don't push"),
    "hard problem of consciousness": ("yes", "no"),
    "human genetic engineering": ("permissible", "impermissible"),
    "immortality": ("yes", "no"),
    "law": ("legal positivism", "legal non-positivism"),
    "laws of nature": ("humean", "non-humean"),
    "logic": ("classical", "non-classical"),
    "mental content": ("externalism", "internalism"),
    "meta-ethics": ("moral realism", "moral anti-realism"),
    "metaontology": ("heavyweight realism", "anti-realism"),
    "metaphilosophy": ("naturalism", "non-naturalism"),
    "method in political philosophy": ("ideal theory", "non-ideal theory"),
    "moral judgment": ("cognitivism", "non-cognitivism"),
    "moral motivation": ("externalism", "internalism"),
    "morality": ("naturalism", "non-naturalism"),
    "ought implies can": ("yes", "no"),
    "rational disagreement": ("permissivism", "non-permissivism"),
    "science": ("scientific realism", "scientific anti-realism"),
    "trolley problem": ("switch", "don't switch"),
}

MERGED_FILES = {
    "human_survey": "merged_human_survey_philosophers_normalized.json",
    "gpt51": "merged_gpt51_philosophers_normalized.json",
    "openai_gpt4o": "merged_openai_gpt4o_philosophers_normalized.json",
    "sonnet45": "merged_sonnet45_philosophers_normalized.json",
    "llama3p18b": "merged_llama3p18b_philosophers_normalized.json",
    "llama3p18b_finetuned": "merged_llama3p18b_finetuned_philosophers_normalized.json",
    "mistral7b": "merged_mistral7b_philosophers_normalized.json",
    "qwen3-4b": "merged_qwen3-4b_philosophers_normalized.json",
}

# Resolve data dirs relative to this script: release/code/*.py → release/data/
import os

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
OUTPUT_DIR = DATA_DIR / "final_normalized_100q"


def normalize_key(key: str) -> str:
    """Remove parenthetical clarifications and normalize whitespace."""
    normalized = re.sub(r"\s*\([^)]*\)", "", key)
    return " ".join(normalized.split())


def extract_stem_and_option(key: str):
    """Split 'stem: option' into (stem, option), both stripped and lowered."""
    norm = normalize_key(key)
    if ":" in norm:
        stem, option = norm.rsplit(":", 1)
        return stem.strip().lower(), option.strip().lower()
    return norm.strip().lower(), norm.strip().lower()


def determine_most_popular_options(human_data):
    """
    For each question stem, count how many human philosophers provided a
    non-null response for each option.  The option with the most responses
    is the 'most popular' (B&C methodology).

    Returns: dict[stem] -> (option_name, original_key_variant)
    """
    # stem -> option -> count of non-null responses
    stem_option_counts = defaultdict(lambda: defaultdict(int))
    # stem -> option -> set of original key variants (to pick the canonical form)
    stem_option_keys = defaultdict(lambda: defaultdict(set))

    for phil in human_data:
        for key, val in phil.get("responses", {}).items():
            if val is None:
                continue
            stem, option = extract_stem_and_option(key)
            stem_option_counts[stem][option] += 1
            stem_option_keys[stem][option].add(normalize_key(key))

    most_popular = {}
    for stem, options in stem_option_counts.items():
        best_option = max(options.items(), key=lambda x: x[1])
        option_name = best_option[0]
        # Pick the canonical key form (there should be just one after normalization)
        canonical_keys = stem_option_keys[stem][option_name]
        canonical_key = sorted(canonical_keys)[0]  # deterministic
        # Store all canonical keys per option (needed for binary positive key lookup)
        all_keys = {opt: stem_option_keys[stem][opt] for opt in options}
        most_popular[stem] = {
            "option": option_name,
            "canonical_key": canonical_key,
            "count": best_option[1],
            "all_options": dict(options),
            "all_keys": all_keys,
        }

    return most_popular


def get_binary_stem_info(stem):
    """Check if stem is a known binary (contradictory pair) stem."""
    if stem not in BINARY_PAIRS:
        return None
    positive, negative = BINARY_PAIRS[stem]
    return {"positive": positive, "negative": negative}


def normalize_dataset(data, most_popular, stems_sorted, binary_stems):
    """
    For each philosopher, extract values:
    - Binary stems: use positive option with complement recovery
    - Multi-option stems: use most-popular option among humans
    """
    normalized = []
    for phil in data:
        new_responses = {}
        # Build a lookup: (stem, option) -> value
        stem_option_vals = {}
        for key, val in phil.get("responses", {}).items():
            if val is None:
                continue
            stem, option = extract_stem_and_option(key)
            stem_option_vals[(stem, option)] = val

        for stem in stems_sorted:
            info = most_popular[stem]

            if stem in binary_stems:
                # Binary stem: renormalize + positive option + complement recovery
                bi = binary_stems[stem]
                positive_val = stem_option_vals.get((stem, bi["positive"]))
                negative_val = stem_option_vals.get((stem, bi["negative"]))

                # Build canonical key for positive option
                canonical_key = info["canonical_key"]
                pos_keys = info.get("all_keys", {}).get(bi["positive"])
                if pos_keys:
                    canonical_key = sorted(pos_keys)[0]
                else:
                    canonical_key = f"{stem}: {bi['positive']}"

                if positive_val is not None and negative_val is not None:
                    # Both sides answered: renormalize to sum=1.0, take positive
                    total = positive_val + negative_val
                    if total > 0:
                        new_responses[canonical_key] = positive_val / total
                    else:
                        new_responses[canonical_key] = 0.5
                elif positive_val is not None:
                    new_responses[canonical_key] = positive_val
                elif negative_val is not None:
                    # Complement recovery: invert the negative value
                    if isinstance(negative_val, (int, float)) and not np.isnan(
                        negative_val
                    ):
                        new_responses[canonical_key] = 1.0 - negative_val
                    else:
                        new_responses[canonical_key] = None
                else:
                    new_responses[canonical_key] = None
            else:
                # Multi-option: use most-popular among humans
                canonical_key = info["canonical_key"]
                option = info["option"]
                val = stem_option_vals.get((stem, option))
                new_responses[canonical_key] = val

        out_phil = {"name": phil.get("name", "Unknown"), "responses": new_responses}
        # Copy metadata
        for k in [
            "url",
            "area_of_specialization",
            "country",
            "gender",
            "target_faculty",
        ]:
            if k in phil:
                out_phil[k] = phil[k]
        normalized.append(out_phil)

    return normalized


def main():
    print("=" * 80)
    print(
        "HYBRID NORMALIZATION: Positive option (binary) + Most Popular (multi-option)"
    )
    print("=" * 80)

    # Load human data first to determine most popular options
    print("\n1. Loading human data to determine most popular options...")
    with open(DATA_DIR / MERGED_FILES["human_survey"]) as f:
        human_data = json.load(f)
    print(f"   Loaded {len(human_data)} human philosophers")

    most_popular = determine_most_popular_options(human_data)
    stems_sorted = sorted(most_popular.keys())
    print(f"   Found {len(stems_sorted)} question stems")

    # Identify binary stems with positive options
    binary_stems = {}
    for stem in stems_sorted:
        bi = get_binary_stem_info(stem)
        if bi is not None:
            binary_stems[stem] = bi

    print(
        f"   Binary stems (positive option + complement recovery): {len(binary_stems)}"
    )
    print(
        f"   Multi-option stems (human most-popular): {len(stems_sorted) - len(binary_stems)}"
    )

    # Show option selection for all stems
    print(f"\n2. Selected options ({len(stems_sorted)} stems):")
    for stem in stems_sorted:
        info = most_popular[stem]
        n_opts = len(info["all_options"])
        if stem in binary_stems:
            bi = binary_stems[stem]
            marker = " [BINARY: positive]"
            print(f"   {stem:45s} -> {bi['positive']:30s}{marker}")
        elif n_opts > 1:
            runner_up = sorted(
                [(k, v) for k, v in info["all_options"].items() if k != info["option"]],
                key=lambda x: -x[1],
            )
            runner_str = (
                f" (runner-up: {runner_up[0][0]}={runner_up[0][1]})"
                if runner_up
                else ""
            )
            print(
                f"   {stem:45s} -> {info['option']:30s} n={info['count']:3d}{runner_str}"
            )
        else:
            print(
                f"   {stem:45s} -> {info['option']:30s} n={info['count']:3d} (only option)"
            )

    # Normalize all datasets
    print(f"\n3. Normalizing all datasets...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, filepath in MERGED_FILES.items():
        print(f"   Processing {name}...")
        with open(DATA_DIR / filepath) as f:
            data = json.load(f)

        normalized = normalize_dataset(data, most_popular, stems_sorted, binary_stems)

        # Count non-null responses
        n_responses = sum(
            1 for p in normalized for v in p["responses"].values() if v is not None
        )
        n_total = len(normalized) * len(stems_sorted)
        pct = n_responses / n_total * 100

        out_path = OUTPUT_DIR / f"{name}_normalized.json"
        with open(out_path, "w") as f:
            json.dump(normalized, f, indent=2)
        print(f"     -> {out_path} ({n_responses}/{n_total} = {pct:.1f}% non-null)")

    # Verify: compare with previous positive-option selection
    print(f"\n4. Comparison with previous 'positive option' selection:")

    # Load old data to compare
    try:
        import subprocess

        result = subprocess.run(
            ["git", "show", "HEAD:final_normalized_100q/human_survey_normalized.json"],
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
        )
        if result.returncode == 0:
            old_data = json.loads(result.stdout)
            old_keys = set()
            for p in old_data:
                old_keys.update(
                    k for k, v in p.get("responses", {}).items() if v is not None
                )
            new_keys = set()
            for p in json.load(open(OUTPUT_DIR / "human_survey_normalized.json")):
                new_keys.update(
                    k for k, v in p.get("responses", {}).items() if v is not None
                )

            old_stems = {extract_stem_and_option(k) for k in old_keys}
            new_stems = {extract_stem_and_option(k) for k in new_keys}

            same = old_stems & new_stems
            diff = old_stems.symmetric_difference(new_stems)
            print(f"   Same stem:option pairs: {len(same)}")
            print(f"   Changed: {len(diff) // 2}")

            # Show what changed
            old_map = {s: o for s, o in old_stems}
            new_map = {s: o for s, o in new_stems}
            changed = [
                (s, old_map.get(s, "?"), new_map.get(s, "?"))
                for s in sorted(set(old_map) | set(new_map))
                if old_map.get(s) != new_map.get(s)
            ]
            for stem, old_opt, new_opt in changed:
                print(f"     {stem:40s} {old_opt:25s} -> {new_opt}")
        else:
            print("   (no git history to compare against)")
    except Exception as e:
        print(f"   (comparison skipped: {e})")

    print(f"\n{'='*80}")
    print("DONE. Run recompute_all_tables.py to regenerate all paper tables.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
