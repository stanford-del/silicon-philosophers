"""
Prompt sensitivity study for Claude Sonnet 4.5 via Anthropic Batch API.

Usage:
    python prompt_sensitivity_sonnet.py submit
    python prompt_sensitivity_sonnet.py status
    python prompt_sensitivity_sonnet.py collect
    python prompt_sensitivity_sonnet.py analyze
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in the environment before running.")
MODEL = "claude-sonnet-4-5-20250929"
# Resolve data dir relative to this script: release/code/*.py → release/data/
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
BATCH_DIR = DATA_DIR / "prompt_sensitivity_batches"

sys.path.insert(0, str(_SCRIPT_DIR))
from prompt_sensitivity_batch import (
    build_persona_prompt_baseline,
    build_question_prompt_variant,
    parse_response_list,
    normalize_option,
    map_to_value,
)


def submit():
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with open(DATA_DIR / "philosophers_with_countries.json") as f:
        philosophers = json.load(f)
    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)
    with open(BATCH_DIR / "sample_meta.json") as f:
        meta = json.load(f)

    sample_indices = meta["sample_indices"]
    sample_phils = [philosophers[i] for i in sample_indices]
    qkeys = list(questions.keys())

    requests = []
    id_map = {}  # sanitized_id -> (phil_idx, qkey)
    for pi, phil_idx in enumerate(sample_indices):
        phil = sample_phils[pi]
        persona = build_persona_prompt_baseline(phil)
        for qi, qkey in enumerate(qkeys):
            question = build_question_prompt_variant(qkey, questions[qkey])
            custom_id = f"p{phil_idx}_q{qi}"
            id_map[custom_id] = {"phil_idx": phil_idx, "qkey": qkey}
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 1024,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": persona + question}],
                    },
                }
            )

    # Save ID mapping for decoding results
    with open(BATCH_DIR / "sonnet45_id_map.json", "w") as f:
        json.dump(id_map, f)

    print(f"Creating batch with {len(requests)} requests for {MODEL}")
    batch = client.beta.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch ID
    ids_path = BATCH_DIR / "batch_ids.json"
    ids = json.load(open(ids_path)) if ids_path.exists() else {}
    ids["sonnet45"] = batch.id
    json.dump(ids, open(ids_path, "w"), indent=2)
    print(f"Saved batch ID to {ids_path}")


def status():
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    ids = json.load(open(BATCH_DIR / "batch_ids.json"))
    bid = ids.get("sonnet45")
    if not bid:
        print("No sonnet45 batch ID found")
        return

    batch = client.beta.messages.batches.retrieve(bid)
    print(f"Batch: {bid}")
    print(f"Status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")


def collect():
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    ids = json.load(open(BATCH_DIR / "batch_ids.json"))
    bid = ids.get("sonnet45")
    if not bid:
        print("No sonnet45 batch ID found")
        return

    batch = client.beta.messages.batches.retrieve(bid)
    if batch.processing_status != "ended":
        print(f"Batch not done yet: {batch.processing_status}")
        print(f"Counts: {batch.request_counts}")
        return

    # Load ID mapping
    with open(BATCH_DIR / "sonnet45_id_map.json") as f:
        id_map = json.load(f)

    out_path = BATCH_DIR / "results_sonnet45.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for result in client.beta.messages.batches.results(bid):
            text = ""
            if result.result.type == "succeeded":
                for block in result.result.message.content:
                    if block.type == "text":
                        text = block.text
                        break
            # Decode sanitized ID back to phil_idx and qkey
            mapping = id_map.get(result.custom_id, {})
            phil_idx = mapping.get("phil_idx", -1)
            qkey = mapping.get("qkey", "")
            orig_id = f"sonnet45__phil{phil_idx}__{qkey}"
            obj = {
                "custom_id": orig_id,
                "response": {
                    "body": {
                        "choices": [{"message": {"content": text}}] if text else []
                    }
                },
            }
            f.write(json.dumps(obj) + "\n")
            count += 1

    print(f"Saved {count} results to {out_path}")


def analyze():
    from scipy import stats

    with open(BATCH_DIR / "sample_meta.json") as f:
        meta = json.load(f)
    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)
    qkeys = list(questions.keys())

    results_path = BATCH_DIR / "results_sonnet45.jsonl"
    if not results_path.exists():
        print("No results file found. Run collect first.")
        return

    # Load baseline
    with open(DATA_DIR / "merged_sonnet45_philosophers_normalized.json") as f:
        baseline_data = json.load(f)

    # Parse variant results
    variant_responses = {}
    n_parsed = 0
    n_failed = 0
    with open(results_path) as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["custom_id"]
            parts = cid.split("__")
            phil_idx = int(parts[1].replace("phil", ""))
            qkey = parts[2]

            choices = obj.get("response", {}).get("body", {}).get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                parsed = parse_response_list(text)
                val = map_to_value(parsed, questions.get(qkey, []))
                variant_responses[(phil_idx, qkey)] = val
                n_parsed += 1
            else:
                n_failed += 1

    print(f"\n{'=' * 50}")
    print(f"MODEL: Sonnet 4.5")
    print(f"{'=' * 50}")
    print(f"  Parsed: {n_parsed}, Failed: {n_failed}")

    baseline_vals = []
    variant_vals = []
    flips = 0
    total_pairs = 0

    for phil_idx in meta["sample_indices"]:
        if phil_idx >= len(baseline_data):
            continue
        bl_responses = baseline_data[phil_idx].get("responses", {})

        for qkey in qkeys:
            bl_val = None
            qkey_lower = qkey.lower()
            for bk, bv in bl_responses.items():
                if (
                    qkey_lower in bk.lower()
                    or bk.lower().split(":")[0].strip() in qkey_lower
                ):
                    bl_val = bv
                    break

            var_val = variant_responses.get((phil_idx, qkey), np.nan)

            if bl_val is not None and not np.isnan(var_val):
                baseline_vals.append(bl_val)
                variant_vals.append(var_val)
                total_pairs += 1
                if abs(bl_val - var_val) > 0.25:
                    flips += 1

    baseline_arr = np.array(baseline_vals)
    variant_arr = np.array(variant_vals)

    if len(baseline_arr) == 0:
        print("  No matching pairs found")
        return

    r, p = stats.pearsonr(baseline_arr, variant_arr)
    print(f"\n  Paired Pearson r:     {r:.4f}  (p={p:.2e})")

    mad = np.mean(np.abs(baseline_arr - variant_arr))
    print(f"  Mean |diff|:          {mad:.4f}")

    rmse = np.sqrt(np.mean((baseline_arr - variant_arr) ** 2))
    print(f"  RMSE (variant diff):  {rmse:.4f}")

    flip_rate = flips / total_pairs if total_pairs > 0 else 0
    print(f"  Flip rate (>0.25):    {flip_rate:.1%}  ({flips}/{total_pairs})")

    t_stat, t_p = stats.ttest_rel(baseline_arr, variant_arr)
    print(f"  Paired t-test:        t={t_stat:.3f}, p={t_p:.4f}")

    bl_mean = np.mean(baseline_arr)
    var_mean = np.mean(variant_arr)
    print(f"  Baseline mean:        {bl_mean:.4f}")
    print(f"  Variant mean:         {var_mean:.4f}")
    print(f"  Mean shift:           {var_mean - bl_mean:+.4f}")

    print(f"  Baseline variance:    {np.var(baseline_arr):.4f}")
    print(f"  Variant variance:     {np.var(variant_arr):.4f}")

    # Per-question flip rate
    q_flips = {}
    for qkey in qkeys:
        for phil_idx in meta["sample_indices"]:
            if phil_idx >= len(baseline_data):
                continue
            bl_resp = baseline_data[phil_idx].get("responses", {})
            bl_val = None
            for bk, bv in bl_resp.items():
                if (
                    qkey.lower() in bk.lower()
                    or bk.lower().split(":")[0].strip() in qkey.lower()
                ):
                    bl_val = bv
                    break
            var_val = variant_responses.get((phil_idx, qkey), np.nan)
            if bl_val is not None and not np.isnan(var_val):
                if abs(bl_val - var_val) > 0.25:
                    q_flips[qkey] = q_flips.get(qkey, 0) + 1
    if q_flips:
        print(f"\n  Top 5 most sensitive questions:")
        for qk, cnt in sorted(q_flips.items(), key=lambda x: -x[1])[:5]:
            print(f"    {qk}: {cnt} flips")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python prompt_sensitivity_sonnet.py [submit|status|collect|analyze]"
        )
        sys.exit(1)
    cmd = sys.argv[1]
    {"submit": submit, "status": status, "collect": collect, "analyze": analyze}[cmd]()
