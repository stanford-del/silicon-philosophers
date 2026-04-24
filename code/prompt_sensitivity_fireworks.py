"""
Prompt sensitivity study for open-source models via Fireworks Batch API.

Uses the SAME 27-philosopher sample (seed=42) and SAME v2 prompt variant
used for GPT-4o, GPT-5.1, and Sonnet 4.5 in prompt_sensitivity_batch.py.

Batch inference = 50% cost savings vs serverless.

Usage:
    source ~/.zshrc
    python prompt_sensitivity_fireworks.py submit     # create datasets + batch jobs
    python prompt_sensitivity_fireworks.py status      # check job progress
    python prompt_sensitivity_fireworks.py collect     # download results
    python prompt_sensitivity_fireworks.py analyze     # compare all 6 models
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy import stats

from prompt_sensitivity_batch import (
    build_persona_prompt_baseline,
    build_question_prompt_variant,
    parse_response_list,
    map_to_value,
)

# ── Config ────────────────────────────────────

API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "whusym")
BASE_URL = "https://api.fireworks.ai/v1"

MODELS = {
    "llama3p18b": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "mistral7b": "accounts/fireworks/models/mistral-7b-instruct-v3",
    "qwen3-4b": "accounts/fireworks/models/qwen3-4b",
}

BASELINE_FILES = {
    "llama3p18b": "merged_llama3p18b_philosophers_normalized.json",
    "mistral7b": "merged_mistral7b_philosophers_normalized.json",
    "qwen3-4b": "merged_qwen3-4b_philosophers_normalized.json",
}

# Resolve data dir relative to this script: release/code/*.py → release/data/
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _SCRIPT_DIR.parent / "data"))
BATCH_DIR = DATA_DIR / "prompt_sensitivity_batches"
FW_DIR = BATCH_DIR / "fireworks"
SAMPLE_META = BATCH_DIR / "sample_meta.json"


def headers(content_type=None):
    h = {"Authorization": f"Bearer {API_KEY}"}
    if content_type:
        h["Content-Type"] = content_type
    return h


# ── Submit ────────────────────────────────────


def submit():
    if not API_KEY:
        print("Set FIREWORKS_API_KEY. Run: source ~/.zshrc")
        sys.exit(1)

    FW_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "philosophers_with_countries.json") as f:
        philosophers = json.load(f)
    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)
    with open(SAMPLE_META) as f:
        meta = json.load(f)

    sample_indices = meta["sample_indices"]
    sample_phils = [philosophers[i] for i in sample_indices]
    qkeys = list(questions.keys())

    print(f"Sample: {len(sample_phils)} philosophers x {len(qkeys)} questions")
    print(f"Total requests per model: {len(sample_phils) * len(qkeys)}")

    job_ids = {}

    for model_key, model_id in MODELS.items():
        print(f"\n--- {model_key} ({model_id}) ---")

        # 1. Build JSONL input
        jsonl_path = FW_DIR / f"batch_input_{model_key}.jsonl"
        with open(jsonl_path, "w") as f:
            for pi, phil_idx in enumerate(sample_indices):
                phil = sample_phils[pi]
                persona = build_persona_prompt_baseline(phil)
                for qkey in qkeys:
                    question = build_question_prompt_variant(qkey, questions[qkey])
                    custom_id = f"{phil_idx}__{qkey}"
                    body = {
                        "messages": [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": question},
                        ],
                        "max_tokens": 150,
                        "temperature": 0.0,
                    }
                    f.write(json.dumps({"custom_id": custom_id, "body": body}) + "\n")
        n_lines = len(sample_indices) * len(qkeys)
        print(f"  Wrote {n_lines} requests to {jsonl_path}")

        # 2. Create input dataset
        input_ds_id = f"prompt-sens-input-{model_key}"
        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/datasets"
        resp = requests.post(
            url,
            headers=headers("application/json"),
            json={
                "datasetId": input_ds_id,
                "dataset": {"userUploaded": {}, "exampleCount": str(n_lines)},
            },
        )
        if resp.status_code == 200:
            print(f"  Created input dataset: {input_ds_id}")
        elif resp.status_code == 409:
            print(f"  Input dataset exists: {input_ds_id}")
        else:
            print(f"  Error creating dataset: {resp.status_code} {resp.text}")
            continue

        # 3. Upload JSONL (skip if already uploaded)
        upload_url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/datasets/{input_ds_id}:upload"
        with open(jsonl_path, "rb") as f:
            resp = requests.post(
                upload_url, headers=headers(), files={"file": (jsonl_path.name, f)}
            )
        if resp.status_code == 200:
            print(f"  Uploaded {jsonl_path.name}")
        elif "already uploaded" in resp.text:
            print(f"  Dataset already uploaded, reusing.")
        else:
            print(f"  Upload error: {resp.status_code} {resp.text}")
            continue

        # 4. Create batch inference job (output dataset is auto-created)
        output_ds_id = f"prompt-sens-output-{model_key}"
        job_url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/batchInferenceJobs"
        job_id = f"prompt-sens-{model_key}"
        payload = {
            "model": model_id,
            "inputDatasetId": f"accounts/{ACCOUNT_ID}/datasets/{input_ds_id}",
            "outputDatasetId": f"accounts/{ACCOUNT_ID}/datasets/{output_ds_id}",
            "inferenceParameters": {
                "maxTokens": 150,
                "temperature": 0.0,
            },
        }
        resp = requests.post(
            job_url,
            headers=headers("application/json"),
            json=payload,
            params={"batchInferenceJobId": job_id},
        )
        if resp.status_code == 200:
            job = resp.json()
            job_name = job.get("name", "")
            print(f"  Batch job created: {job_name}")
            job_ids[model_key] = job_name
        else:
            print(f"  Job creation error: {resp.status_code} {resp.text}")

    # Save job info
    with open(FW_DIR / "batch_job_ids.json", "w") as f:
        json.dump(job_ids, f, indent=2)
    print(f"\nJob IDs saved to {FW_DIR / 'batch_job_ids.json'}")
    print("Run: python prompt_sensitivity_fireworks.py status")


# ── Status ────────────────────────────────────


def check_status():
    if not API_KEY:
        print("Set FIREWORKS_API_KEY.")
        sys.exit(1)

    with open(FW_DIR / "batch_job_ids.json") as f:
        job_ids = json.load(f)

    for model_key, job_name in job_ids.items():
        job_id = job_name.split("/")[-1]
        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/batchInferenceJobs/{job_id}"
        resp = requests.get(url, headers=headers())
        if resp.status_code == 200:
            job = resp.json()
            state = job.get("state", "UNKNOWN")
            progress = job.get("jobProgress") or {}
            done = progress.get("completedRequests", "?")
            total = progress.get("totalRequests", "?")
            failed = progress.get("failedRequests", 0)
            print(
                f"  {model_key}: state={state}  completed={done}/{total}  failed={failed}"
            )
        else:
            print(f"  {model_key}: error {resp.status_code} {resp.text}")


# ── Collect ───────────────────────────────────


def collect():
    if not API_KEY:
        print("Set FIREWORKS_API_KEY.")
        sys.exit(1)

    with open(FW_DIR / "batch_job_ids.json") as f:
        job_ids = json.load(f)

    for model_key, job_name in job_ids.items():
        job_id = job_name.split("/")[-1]
        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/batchInferenceJobs/{job_id}"
        resp = requests.get(url, headers=headers())
        if resp.status_code != 200:
            print(f"  {model_key}: error {resp.status_code}")
            continue
        job = resp.json()
        state = job.get("state", "")
        if "COMPLETED" not in state:
            print(f"  {model_key}: not done yet (state={state})")
            continue

        # Download output dataset
        output_ds_id = f"prompt-sens-output-{model_key}"
        dl_url = (
            f"{BASE_URL}/accounts/{ACCOUNT_ID}/datasets/"
            f"{output_ds_id}:getDownloadEndpoint"
        )
        resp = requests.get(dl_url, headers=headers())
        if resp.status_code != 200:
            print(
                f"  {model_key}: download endpoint error {resp.status_code} {resp.text}"
            )
            continue

        dl_info = resp.json()
        signed_urls = dl_info.get("filenameToSignedUrls", {})
        for fname, signed_url in signed_urls.items():
            basename = Path(fname).name
            out_path = FW_DIR / f"results_{model_key}_{basename}"
            r = requests.get(signed_url)
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"  {model_key}: saved {out_path} ({len(r.content)} bytes)")

        # Also try to parse and save as our standard format
        _convert_results(model_key)


def _convert_results(model_key):
    """Convert batch output JSONL to our standard {phil_idx__qkey: text} format."""
    result_files = list(FW_DIR.glob(f"results_{model_key}_*.jsonl"))
    if not result_files:
        # Try without extension filter
        result_files = list(FW_DIR.glob(f"results_{model_key}_*"))
    combined = {}
    for rf in result_files:
        try:
            with open(rf) as f:
                for line in f:
                    obj = json.loads(line)
                    cid = obj.get("custom_id", "")
                    resp = obj.get("response", obj.get("body", {}))
                    choices = resp.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                        combined[cid] = text
        except Exception as e:
            print(f"    Warning parsing {rf}: {e}")
    if combined:
        out = FW_DIR / f"results_{model_key}.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"  {model_key}: consolidated {len(combined)} responses -> {out}")


# ── Analyze ───────────────────────────────────


def analyze():
    with open(SAMPLE_META) as f:
        meta = json.load(f)
    with open(DATA_DIR / "question_answer_options.json") as f:
        questions = json.load(f)
    qkeys = list(questions.keys())

    print("=" * 70)
    print("PROMPT SENSITIVITY: ALL 6 NON-FINE-TUNED MODELS")
    print("=" * 70)
    print(f"Sample: {meta['n_sample']} philosophers, {len(qkeys)} questions")
    print(f"Variant: No PhilPapers context (direct philosophical question)\n")

    all_stats = {}

    # GPT/Sonnet from existing batch results
    all_stats.update(_load_gpt_sonnet_stats(questions, meta, qkeys))

    # Open-source from Fireworks batch results
    for model_key in MODELS:
        results_file = FW_DIR / f"results_{model_key}.json"
        baseline_rel = BASELINE_FILES.get(model_key)
        baseline_file = DATA_DIR / baseline_rel if baseline_rel else None

        if not results_file.exists():
            print(f"{model_key}: no results. Run 'collect' first.")
            continue
        if baseline_file is None or not baseline_file.exists():
            print(f"{model_key}: no baseline at {baseline_file}")
            continue

        with open(results_file) as f:
            raw_results = json.load(f)
        with open(baseline_file) as f:
            baseline_data = json.load(f)

        # Parse variant responses
        variant_responses = {}
        for key, text in raw_results.items():
            parts = key.split("__", 1)
            phil_idx = int(parts[0])
            qkey = parts[1]
            if text:
                parsed = parse_response_list(text)
                val = map_to_value(parsed, questions.get(qkey, []))
                variant_responses[(phil_idx, qkey)] = val

        s = _compute_stats(
            model_key, variant_responses, baseline_data, meta, qkeys, questions
        )
        if s:
            all_stats[model_key] = s

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE: Prompt Sensitivity Across All 6 Models")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'r':>8} {'MAD':>8} {'RMSE':>8} {'Flip%':>8} {'Shift':>8}")
    print("-" * 60)
    for mk in ["gpt-4o", "gpt-5.1", "sonnet45", "llama3p18b", "mistral7b", "qwen3-4b"]:
        if mk in all_stats:
            s = all_stats[mk]
            print(
                f"{mk:<20} {s['r']:>8.3f} {s['mad']:>8.4f} {s['rmse']:>8.4f} "
                f"{s['flip_rate']:>7.1%} {s['mean_shift']:>+8.4f}"
            )

    out_file = FW_DIR / "prompt_sensitivity_all_6_models.json"
    with open(out_file, "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_stats.items()},
            f,
            indent=2,
        )
    print(f"\nSaved to {out_file}")


def _compute_stats(model_key, variant_responses, baseline_data, meta, qkeys, questions):
    """Compute prompt sensitivity stats for one model."""
    bl_vals, var_vals = [], []
    flips = 0
    q_flips = {}

    for phil_idx in meta["sample_indices"]:
        if phil_idx >= len(baseline_data):
            continue
        bl_resp = baseline_data[phil_idx].get("responses", {})
        for qkey in qkeys:
            bl_val = _match_baseline(bl_resp, qkey)
            var_val = variant_responses.get((phil_idx, qkey), np.nan)
            if bl_val is not None and not np.isnan(var_val):
                bl_vals.append(bl_val)
                var_vals.append(var_val)
                if abs(bl_val - var_val) > 0.25:
                    flips += 1
                    q_flips[qkey] = q_flips.get(qkey, 0) + 1

    if len(bl_vals) < 10:
        print(f"  {model_key}: too few pairs ({len(bl_vals)})")
        return None

    ba, va = np.array(bl_vals), np.array(var_vals)
    r, _ = stats.pearsonr(ba, va)
    mad = np.mean(np.abs(ba - va))
    rmse = np.sqrt(np.mean((ba - va) ** 2))
    flip_rate = flips / len(bl_vals)
    t_stat, t_p = stats.ttest_rel(ba, va)

    print(f"\n{'='*50}")
    print(f"MODEL: {model_key}")
    print(f"{'='*50}")
    print(f"  Pairs: {len(bl_vals)}")
    print(f"  Pearson r:     {r:.4f}")
    print(f"  MAD:           {mad:.4f}")
    print(f"  RMSE:          {rmse:.4f}")
    print(f"  Flip rate:     {flip_rate:.1%} ({flips}/{len(bl_vals)})")
    print(f"  Mean shift:    {np.mean(va) - np.mean(ba):+.4f}")
    print(f"  t-test:        t={t_stat:.3f}, p={t_p:.4f}")

    if q_flips:
        print(f"  Top 5 sensitive questions:")
        for qk, cnt in sorted(q_flips.items(), key=lambda x: -x[1])[:5]:
            print(f"    {qk}: {cnt} flips")

    return {
        "r": r,
        "mad": mad,
        "rmse": rmse,
        "flip_rate": flip_rate,
        "t_stat": t_stat,
        "t_p": t_p,
        "mean_shift": float(np.mean(va) - np.mean(ba)),
        "bl_var": float(np.var(ba)),
        "var_var": float(np.var(va)),
        "n_pairs": len(bl_vals),
    }


def _match_baseline(bl_resp, qkey):
    """Match question key to baseline response dict."""
    qkey_lower = qkey.lower()
    for bk, bv in bl_resp.items():
        bk_stem = bk.lower().split(":")[0].strip()
        if qkey_lower in bk.lower() or bk_stem in qkey_lower or qkey_lower in bk_stem:
            return bv
    return None


def _load_gpt_sonnet_stats(questions, meta, qkeys):
    """Compute stats for GPT/Sonnet from existing batch results."""
    result_stats = {}
    pairs = [
        (
            "gpt-4o",
            "results_gpt-4o.jsonl",
            "merged_openai_gpt4o_philosophers_normalized.json",
        ),
        (
            "gpt-5.1",
            "results_gpt-5.1.jsonl",
            "merged_gpt51_philosophers_normalized.json",
        ),
        (
            "sonnet45",
            "results_sonnet45.jsonl",
            "merged_sonnet45_philosophers_normalized.json",
        ),
    ]
    for model_key, results_fname, baseline_fname in pairs:
        results_path = BATCH_DIR / results_fname
        baseline_path = DATA_DIR / baseline_fname
        if not results_path.exists() or not baseline_path.exists():
            continue
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        variant_responses = {}
        with open(results_path) as f:
            for line in f:
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                if "__phil" in cid:
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
                elif cid.startswith("p") and "_q" in cid:
                    parts = cid.split("_")
                    phil_idx = int(parts[0][1:])
                    qi = int(parts[1][1:])
                    if qi < len(qkeys):
                        qkey = qkeys[qi]
                        result = obj.get("result", {})
                        msg = result.get("message", {})
                        text = ""
                        if isinstance(msg.get("content"), list):
                            for block in msg["content"]:
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    break
                        elif isinstance(msg.get("content"), str):
                            text = msg["content"]
                        if text:
                            parsed = parse_response_list(text)
                            val = map_to_value(parsed, questions.get(qkey, []))
                            variant_responses[(phil_idx, qkey)] = val

        s = _compute_stats(
            model_key, variant_responses, baseline_data, meta, qkeys, questions
        )
        if s:
            result_stats[model_key] = s

    return result_stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python prompt_sensitivity_fireworks.py [submit|status|collect|analyze]"
        )
        sys.exit(1)
    cmd = sys.argv[1]
    {"submit": submit, "status": check_status, "collect": collect, "analyze": analyze}[
        cmd
    ]()
