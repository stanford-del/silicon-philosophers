"""
Prompt Sensitivity Study — OpenAI Batch API

Sends 10% of philosophers (28 random, seeded) × all 100 questions
to GPT-5.1 and GPT-4o with an ALTERNATIVE framing (no PhilPapers context,
direct philosophical question) and compares with existing baseline results.

Uses OpenAI Batch API for 50% cost savings.

Usage:
    python prompt_sensitivity_batch.py submit     # Create and submit batches
    python prompt_sensitivity_batch.py status      # Check batch status
    python prompt_sensitivity_batch.py collect     # Download results when done
    python prompt_sensitivity_batch.py analyze     # Compare with baseline
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SEED = 42
SAMPLE_FRAC = 0.10  # 10% of 277 = 28 philosophers
MODELS = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-5.1": "gpt-5.1-2025-11-13",
}
# Resolve data dir relative to this script: release/code/*.py → release/data/
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
BATCH_DIR = DATA_DIR / "prompt_sensitivity_batches"
BATCH_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_persona_prompt_baseline(p):
    """Original prompt (baseline) — same as model_eval.py."""
    prompt = "You are a professional philosopher"
    inst = p.get("current_institution")
    country = p.get("current_institution_country")
    if inst:
        prompt += f" at {inst}"
        if country:
            prompt += f" ({country})"
    prompt += ".\n\n"
    phd = p.get("phd_institution")
    phd_c = p.get("phd_country")
    phd_y = p.get("year_of_phd_degree")
    if phd or phd_c or phd_y:
        prompt += "Your Educational Background:\n"
        if phd:
            prompt += f"- PhD from {phd}"
            if phd_c:
                prompt += f" ({phd_c})"
            if phd_y:
                prompt += f" in {phd_y}"
            prompt += "\n"
        prompt += "\n"
    specs = p.get("areas_of_specialization", [])
    if specs:
        prompt += "Your Areas of Specialization:\n"
        for s in specs[:5]:
            prompt += f"- {s}\n"
        prompt += "\n"
    interests = p.get("areas_of_interest", [])
    if interests:
        prompt += "Your Areas of Interest:\n"
        for i in interests[:8]:
            prompt += f"- {i}\n"
        prompt += "\n"
    return prompt


def build_question_prompt_baseline(qkey, options):
    """Original question prompt (baseline)."""
    prompt = (
        "You are answering a survey by the reputable website PhilPapers, "
        "which collects responses across various philosophical domains based "
        "on your background. You may select one or multiple options if you "
        "deem it appropriate to accurately represent your position.\n\n"
        "Please respond with your chosen option(s) as a JSON list. If "
        "selecting multiple options, ensure they are logically consistent.\n\n"
        "Examples of valid responses:\n"
        '- ["Accept: physicalism"]\n'
        '- ["Accept: physicalism", "Reject: non-physicalism"]\n'
        '- ["Lean towards: physicalism", "Lean against: non-physicalism"]\n\n'
        "Given your philosophical profile above, please express your stance "
        "on the following question.\n\n"
    )
    prompt += f"Question: {qkey}\n\nAvailable response options:\n"
    for o in options:
        prompt += f"- {o}\n"
    prompt += "\nRespond ONLY with the JSON list, no other text or explanation.\n\n"
    return prompt


def build_question_prompt_variant(qkey, options):
    """Alternative framing — NO PhilPapers context, direct question."""
    prompt = (
        "Based on your philosophical expertise and training, what is your "
        "considered position on the following question?\n\n"
        "Respond with your chosen option(s) as a JSON list.\n\n"
        "Examples:\n"
        '- ["Accept: physicalism"]\n'
        '- ["Lean towards: physicalism"]\n\n'
    )
    prompt += f"Question: {qkey}\n\nOptions:\n"
    for o in options:
        prompt += f"- {o}\n"
    prompt += "\nRespond ONLY with the JSON list.\n"
    return prompt


# ---------------------------------------------------------------------------
# Batch file creation
# ---------------------------------------------------------------------------


def create_batch_files():
    """Create JSONL batch files for both models."""
    with open(DATA_DIR / "philosophers_with_countries.json") as f:
        philosophers = json.load(f)
    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)

    random.seed(SEED)
    n_sample = max(1, int(len(philosophers) * SAMPLE_FRAC))
    sample_indices = sorted(random.sample(range(len(philosophers)), n_sample))
    sample_phils = [philosophers[i] for i in sample_indices]

    print(f"Sampled {len(sample_phils)} philosophers (seed={SEED})")
    print(f"Questions: {len(questions)}")
    print(f"Total requests per model: {len(sample_phils) * len(questions)}")

    # Save sample indices for later analysis
    meta = {
        "seed": SEED,
        "n_sample": len(sample_phils),
        "sample_indices": sample_indices,
        "sample_names": [p["name"] for p in sample_phils],
        "n_questions": len(questions),
    }
    with open(BATCH_DIR / "sample_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for model_key, model_id in MODELS.items():
        requests = []
        for pi, phil in enumerate(sample_phils):
            persona = build_persona_prompt_baseline(phil)
            for qkey, options in questions.items():
                question_prompt = build_question_prompt_variant(qkey, options)
                full_user = persona + question_prompt
                custom_id = f"{model_key}__phil{sample_indices[pi]}__{qkey}"
                body = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": full_user}],
                    "temperature": 0.0,
                }
                # GPT-5+ models require max_completion_tokens
                if "gpt-5" in model_id:
                    body["max_completion_tokens"] = 150
                else:
                    body["max_tokens"] = 150
                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                requests.append(req)

        out_path = BATCH_DIR / f"batch_{model_key}.jsonl"
        with open(out_path, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(requests)} requests to {out_path}")

    return meta


# ---------------------------------------------------------------------------
# Submit batches
# ---------------------------------------------------------------------------


def submit_batches():
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    meta = create_batch_files()
    batch_ids = {}

    for model_key in MODELS:
        jsonl_path = BATCH_DIR / f"batch_{model_key}.jsonl"

        # Upload file
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"  Uploaded {jsonl_path} → file_id={file_obj.id}")

        # Create batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"prompt_sensitivity_{model_key}"},
        )
        batch_ids[model_key] = batch.id
        print(f"  Created batch {batch.id} for {model_key} (status: {batch.status})")

    with open(BATCH_DIR / "batch_ids.json", "w") as f:
        json.dump(batch_ids, f, indent=2)
    print(f"\nBatch IDs saved to {BATCH_DIR / 'batch_ids.json'}")
    print("Run `python prompt_sensitivity_batch.py status` to check progress.")


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


def check_status():
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(BATCH_DIR / "batch_ids.json") as f:
        batch_ids = json.load(f)

    for model_key, bid in batch_ids.items():
        b = client.batches.retrieve(bid)
        done = b.request_counts.completed if b.request_counts else "?"
        total = b.request_counts.total if b.request_counts else "?"
        failed = b.request_counts.failed if b.request_counts else "?"
        print(
            f"  {model_key}: status={b.status}  completed={done}/{total}  failed={failed}"
        )
        if b.status == "completed":
            print(f"    → output_file_id={b.output_file_id}")


# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------


def collect_results():
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(BATCH_DIR / "batch_ids.json") as f:
        batch_ids = json.load(f)

    for model_key, bid in batch_ids.items():
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(f"  {model_key}: not done yet (status={b.status}), skipping")
            continue

        content = client.files.content(b.output_file_id)
        out_path = BATCH_DIR / f"results_{model_key}.jsonl"
        with open(out_path, "wb") as f:
            f.write(content.content)
        print(f"  {model_key}: saved {out_path}")

    print("\nRun `python prompt_sensitivity_batch.py analyze` to compare.")


# ---------------------------------------------------------------------------
# Parse & analyze
# ---------------------------------------------------------------------------


def parse_response_list(text):
    """Parse LLM JSON list response."""
    import re

    text = text.strip()
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted
    return [text]


def normalize_option(opt):
    """Normalize option string for matching."""
    s = str(opt).lower().strip().rstrip(".,!?;")
    s = " ".join(s.split())
    s = s.replace(" :", ":").replace(":", ": ").replace(":  ", ": ")
    return " ".join(s.split())


def map_to_value(responses, options):
    """Map parsed responses to [0,1] value using same logic as normalization pipeline."""
    stances = {
        "accept": 1.0,
        "lean towards": 0.75,
        "lean toward": 0.75,
        "neutral towards": 0.5,
        "neutral toward": 0.5,
        "agnostic": 0.5,
        "lean against": 0.25,
        "reject": 0.0,
    }
    vals = []
    for r in responses:
        rn = normalize_option(r)
        for stance_key, stance_val in stances.items():
            if rn.startswith(stance_key + ": "):
                vals.append(stance_val)
                break
    return np.mean(vals) if vals else np.nan


def analyze():
    """Compare variant results with baseline."""
    from scipy import stats

    with open(BATCH_DIR / "sample_meta.json") as f:
        meta = json.load(f)

    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)
    qkeys = list(questions.keys())

    # Load baseline data
    baselines = {}
    for model_key, fn in [
        ("gpt-4o", "merged_openai_gpt4o_philosophers_normalized.json"),
        ("gpt-5.1", "merged_gpt51_philosophers_normalized.json"),
    ]:
        with open(DATA_DIR / fn) as f:
            data = json.load(f)
        baselines[model_key] = data

    print("=" * 70)
    print("PROMPT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Sample: {meta['n_sample']} philosophers, {meta['n_questions']} questions")
    print(f"Variant: No PhilPapers context (direct philosophical question)")
    print()

    for model_key in MODELS:
        results_path = BATCH_DIR / f"results_{model_key}.jsonl"
        if not results_path.exists():
            print(f"  {model_key}: no results file found, skipping")
            continue

        # Parse batch results
        variant_responses = {}  # (phil_idx, qkey) -> value
        n_parsed = 0
        n_failed = 0
        with open(results_path) as f:
            for line in f:
                obj = json.loads(line)
                cid = obj["custom_id"]
                parts = cid.split("__")
                phil_idx = int(parts[1].replace("phil", ""))
                qkey = parts[2]

                body = obj.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    parsed = parse_response_list(text)
                    val = map_to_value(parsed, questions.get(qkey, []))
                    variant_responses[(phil_idx, qkey)] = val
                    n_parsed += 1
                else:
                    n_failed += 1

        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_key}")
        print(f"{'=' * 50}")
        print(f"  Parsed: {n_parsed}, Failed: {n_failed}")

        # Build baseline values for same philosophers
        baseline_data = baselines.get(model_key, [])
        baseline_vals = []
        variant_vals = []
        flips = 0
        total_pairs = 0

        for phil_idx in meta["sample_indices"]:
            if phil_idx >= len(baseline_data):
                continue
            bl_responses = baseline_data[phil_idx].get("responses", {})

            for qkey in qkeys:
                # Normalize question key for baseline lookup
                bl_val = None
                for bk, bv in bl_responses.items():
                    if bk.split(":")[0].strip().lower() == qkey.lower().replace(
                        " ", ""
                    ).replace("-", ""):
                        bl_val = bv
                        break
                # Try exact substring match
                if bl_val is None:
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
                    # "Flip" = categorical change (>0.25 difference on 0-1 scale)
                    if abs(bl_val - var_val) > 0.25:
                        flips += 1

        baseline_arr = np.array(baseline_vals)
        variant_arr = np.array(variant_vals)

        if len(baseline_arr) == 0:
            print("  No matching pairs found — check question key normalization")
            continue

        # --- Metrics ---
        # 1. Paired correlation
        r, p = stats.pearsonr(baseline_arr, variant_arr)
        print(f"\n  Paired Pearson r:     {r:.4f}  (p={p:.2e})")

        # 2. Mean absolute difference
        mad = np.mean(np.abs(baseline_arr - variant_arr))
        print(f"  Mean |diff|:          {mad:.4f}")

        # 3. RMSE between variants
        rmse = np.sqrt(np.mean((baseline_arr - variant_arr) ** 2))
        print(f"  RMSE (variant diff):  {rmse:.4f}")

        # 4. Flip rate
        flip_rate = flips / total_pairs if total_pairs > 0 else 0
        print(f"  Flip rate (>0.25):    {flip_rate:.1%}  ({flips}/{total_pairs})")

        # 5. Paired t-test on means
        t_stat, t_p = stats.ttest_rel(baseline_arr, variant_arr)
        print(f"  Paired t-test:        t={t_stat:.3f}, p={t_p:.4f}")

        # 6. Mean shift
        bl_mean = np.mean(baseline_arr)
        var_mean = np.mean(variant_arr)
        print(f"  Baseline mean:        {bl_mean:.4f}")
        print(f"  Variant mean:         {var_mean:.4f}")
        print(f"  Mean shift:           {var_mean - bl_mean:+.4f}")

        # 7. Variance comparison
        bl_var = np.var(baseline_arr)
        var_var = np.var(variant_arr)
        print(f"  Baseline variance:    {bl_var:.4f}")
        print(f"  Variant variance:     {var_var:.4f}")

        # 8. Per-question flip rate (top 5)
        q_flips = {}
        for qkey in qkeys:
            q_bl = []
            q_var = []
            for phil_idx in meta["sample_indices"]:
                if phil_idx >= len(baseline_data):
                    continue
                bl_resp = baseline_data[phil_idx].get("responses", {})
                bl_val = None
                qkey_lower = qkey.lower()
                for bk, bv in bl_resp.items():
                    if (
                        qkey_lower in bk.lower()
                        or bk.lower().split(":")[0].strip() in qkey_lower
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python prompt_sensitivity_batch.py [submit|status|collect|analyze]"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "submit":
        submit_batches()
    elif cmd == "status":
        check_status()
    elif cmd == "collect":
        collect_results()
    elif cmd == "analyze":
        analyze()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
