"""
Fireworks AI: End-to-End Philosopher Survey Pipeline

Reproduces the paper's full methodology via Fireworks API:
  1. collect   - Query stock/finetuned models for all 277 philosophers x 100 questions
  2. normalize - Parse raw responses, code to [0,1], produce 277x100 normalized matrix
  3. analyze   - Compute all paper metrics (KL/JS, Mantel, PCA, demographics)
  4. convert   - Convert local DPO data to Fireworks format
  5. upload    - Upload DPO dataset to Fireworks
  6. train     - Launch DPO fine-tuning job
  7. deploy    - Deploy a fine-tuned model

Usage:
    export FIREWORKS_API_KEY="your-key-here"
    export FIREWORKS_ACCOUNT_ID="your-account-id"  # only needed for DPO steps

    # End-to-end baseline:
    python fireworks_dpo_finetune.py collect --models llama3p18b
    python fireworks_dpo_finetune.py normalize --models llama3p18b
    python fireworks_dpo_finetune.py analyze --models llama3p18b

    # Or all 3 stock models:
    python fireworks_dpo_finetune.py collect
    python fireworks_dpo_finetune.py normalize
    python fireworks_dpo_finetune.py analyze

    # DPO pipeline:
    python fireworks_dpo_finetune.py convert
    python fireworks_dpo_finetune.py upload
    python fireworks_dpo_finetune.py train
    python fireworks_dpo_finetune.py deploy
    python fireworks_dpo_finetune.py collect --models llama3p18b-dpo
    python fireworks_dpo_finetune.py normalize --models llama3p18b-dpo
    python fireworks_dpo_finetune.py analyze

    # SFT pipeline (same data, reformatted):
    python fireworks_dpo_finetune.py sft-convert
    python fireworks_dpo_finetune.py sft-upload
    python fireworks_dpo_finetune.py sft-train
    python fireworks_dpo_finetune.py deploy --model accounts/<id>/models/philosopher-llama3p1-8b-sft
    python fireworks_dpo_finetune.py collect --models llama3p18b-sft
    python fireworks_dpo_finetune.py normalize --models llama3p18b-sft
    python fireworks_dpo_finetune.py analyze
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
BASE_URL = "https://api.fireworks.ai/v1"

# Models matching the paper (serverless on Fireworks)
BASELINE_MODELS = {
    "llama3p18b": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "mistral7b": "accounts/fireworks/models/mistral-7b-instruct-v3",
    "qwen3-4b": "accounts/fireworks/models/qwen3-4b",
}

# Fine-tuning base model
FT_BASE_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"

# DPO config
DPO_DATASET_ID = "philosopher-dpo-data"
DPO_OUTPUT_MODEL_ID = "philosopher-llama3p1-8b-dpo"
DPO_JOB_ID = "philosopher-dpo-job"
DPO_INPUT_JSONL = "improved_dpo_data/philosopher_dpo_train_FINAL.jsonl"
DPO_CONVERTED_JSONL = "improved_dpo_data/philosopher_dpo_fireworks_format.jsonl"
DPO_TRAINING_CONFIG = {
    "epochs": 3,
    "learningRate": 2e-5,
    "batchSize": 8,
    "loraRank": 16,
}
DPO_LOSS_CONFIG = {"lossMethod": "DPO"}

# SFT config (same source data, reformatted — only the "chosen" response is used)
SFT_DATASET_ID = "philosopher-sft-data"
SFT_OUTPUT_MODEL_ID = "philosopher-llama3p1-8b-sft"
SFT_JOB_ID = "philosopher-sft-job"
SFT_CONVERTED_JSONL = "improved_dpo_data/philosopher_sft_fireworks_format.jsonl"
SFT_TRAINING_CONFIG = {
    "epochs": 3,
    "learningRate": 2e-5,
    "batchSize": 8,
    "loraRank": 16,
}

# Data files
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
QUESTIONS_FILE = "question_answer_options.json"
HUMAN_NORMALIZED = "final_normalized_100q/human_survey_normalized.json"

# Directories
RAW_DIR = "fireworks_raw_responses"
NORM_DIR = "fireworks_normalized_100q"

# Eval settings
MAX_TOKENS = 150
TEMPERATURE = 0.0
MAX_RETRIES = 3
TEST_LIMIT = None  # Set to e.g. 5 for quick testing

# Response coding: attitude -> [0, 1] ordinal scale
ATTITUDE_SCORES = {
    "accept": 1.0,
    "lean towards": 0.75,
    "neutral towards": 0.5,
    "lean against": 0.25,
    "reject": 0.0,
    # Aliases
    "lean toward": 0.75,
    "neutral": 0.5,
    "agnostic/undecided": 0.5,
    "agnostic": 0.5,
}

# Binary stems: positive -> negative option (for complement recovery)
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

N_BINS = 20  # Paper uses 20 bins for KL/JS
MANTEL_PERMS = 999


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def headers(content_type=None):
    h = {"Authorization": f"Bearer {API_KEY}"}
    if content_type:
        h["Content-Type"] = content_type
    return h


def get_client():
    return OpenAI(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")


def resolve_models(args_models):
    """Return list of (label, model_id) tuples from --models arg."""
    all_models = dict(BASELINE_MODELS)
    if ACCOUNT_ID:
        all_models["llama3p18b-dpo"] = (
            f"accounts/{ACCOUNT_ID}/models/{DPO_OUTPUT_MODEL_ID}"
        )
        all_models["llama3p18b-sft"] = (
            f"accounts/{ACCOUNT_ID}/models/{SFT_OUTPUT_MODEL_ID}"
        )
    if args_models:
        selected = args_models.split(",")
    else:
        selected = list(BASELINE_MODELS.keys())
    result = []
    for label in selected:
        if label in all_models:
            result.append((label, all_models[label]))
        else:
            print(f"Unknown model '{label}'. Available: {list(all_models.keys())}")
    return result


# ──────────────────────────────────────────────
# Step 1: COLLECT — query models via Fireworks API
# ──────────────────────────────────────────────


def build_persona_prompt(persona):
    """Build second-person persona prompt from philosopher data (matches model_eval.py)."""
    specializations = persona.get("areas_of_specialization", [])
    interests = persona.get("areas_of_interest", [])
    phd = persona.get("phd_institution")
    phd_country = persona.get("phd_country")
    phd_year = persona.get("year_of_phd_degree")
    institution = persona.get("current_institution")
    institution_country = persona.get("current_institution_country")

    prompt = "You are a professional philosopher"
    if institution:
        prompt += f" at {institution}"
        if institution_country:
            prompt += f" ({institution_country})"
    prompt += ".\n\n"

    if phd or phd_country or phd_year:
        prompt += "Your Educational Background:\n"
        if phd:
            prompt += f"- PhD from {phd}"
            if phd_country:
                prompt += f" ({phd_country})"
            if phd_year:
                prompt += f" in {phd_year}"
            prompt += "\n"
        prompt += "\n"

    if specializations:
        prompt += "Your Areas of Specialization:\n"
        for spec in specializations[:5]:
            prompt += f"- {spec}\n"
        prompt += "\n"

    if interests:
        prompt += "Your Areas of Interest:\n"
        for interest in interests[:8]:
            prompt += f"- {interest}\n"
        prompt += "\n"

    return prompt


def build_question_prompt(question_key, answer_options, prompt_variant="paper"):
    """Build survey question prompt. Variants: 'paper' (original), 'v2' (simplified)."""
    if prompt_variant == "v2":
        # Same alternate prompt used for GPT-4o, GPT-5.1, Sonnet 4.5
        # (from prompt_sensitivity_batch.py build_question_prompt_variant)
        # No PhilPapers context, direct philosophical question
        prompt = (
            "Based on your philosophical expertise and training, what is your "
            "considered position on the following question?\n\n"
            "Respond with your chosen option(s) as a JSON list.\n\n"
            "Examples:\n"
            '- ["Accept: physicalism"]\n'
            '- ["Lean towards: physicalism"]\n\n'
        )
        prompt += f"Question: {question_key}\n\nOptions:\n"
        for option in answer_options:
            prompt += f"- {option}\n"
        prompt += "\nRespond ONLY with the JSON list.\n"
    else:
        # Original paper prompt (matches model_eval.py)
        prompt = (
            "You are answering a survey by the reputable website PhilPapers, which collects\n"
            "responses across various philosophical domains based on your background. "
            "You may select one or multiple options if you deem it appropriate to accurately "
            "represent your position.\n\n"
            "Please respond with your chosen option(s) as a JSON list. If selecting multiple "
            "options, ensure they are logically consistent.\n\n"
            "Examples of valid responses:\n"
            '- ["Accept: physicalism"]\n'
            '- ["Accept: physicalism", "Reject: non-physicalism"]\n'
            '- ["Lean towards: physicalism", "Lean against: non-physicalism"]\n\n'
            "Given your philosophical profile above, please express your stance on the "
            "following question.\n\n"
            f"Question: {question_key}\n\n"
            "Available response options:\n"
        )
        for option in answer_options:
            prompt += f"- {option}\n"
        prompt += "\nRespond ONLY with the JSON list, no other text or explanation.\n\n"
    return prompt


def parse_response_list(text):
    """Parse LLM response to extract JSON list of option strings."""
    if not text:
        return []
    text = text.strip()
    # Try JSON extraction
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass
    # Fallback: quoted strings
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted
    return [text]


def call_model(client, model_id, system_prompt, user_prompt, retries=MAX_RETRIES):
    """Chat completion call with retries on transient errors."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"    API error after {retries} attempts: {e}")
                return None


def cmd_collect(args):
    """Query Fireworks models for all philosopher x question pairs."""
    if not API_KEY:
        print("Set FIREWORKS_API_KEY environment variable.")
        sys.exit(1)

    with open(PHILOSOPHERS_FILE) as f:
        philosophers = json.load(f)
    with open(QUESTIONS_FILE) as f:
        question_options = json.load(f)

    models = resolve_models(args.models)
    client = get_client()
    prompt_variant = getattr(args, "prompt", "paper")

    phil_list = philosophers[:TEST_LIMIT] if TEST_LIMIT else philosophers

    for label, model_id in models:
        # Include prompt variant in dir name so runs don't collide
        dir_suffix = f"_{prompt_variant}" if prompt_variant != "paper" else ""
        out_dir = os.path.join(RAW_DIR, f"{label}{dir_suffix}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nCollecting: {label} ({model_id})  [prompt={prompt_variant}]")
        print(f"  {len(phil_list)} philosophers x {len(question_options)} questions")

        for pi, philosopher in enumerate(phil_list):
            name = philosopher["name"]
            safe_name = name.replace(" ", "_").replace("/", "_")
            out_file = os.path.join(out_dir, f"{safe_name}.json")

            if os.path.exists(out_file):
                continue

            system_prompt = build_persona_prompt(philosopher)
            responses = {}

            for q_key, options in question_options.items():
                user_prompt = build_question_prompt(q_key, options, prompt_variant)
                raw = call_model(client, model_id, system_prompt, user_prompt)
                parsed = parse_response_list(raw) if raw else []
                responses[q_key] = {
                    "raw_response": raw,
                    "parsed": parsed,
                    "options_shown": options,
                }

            result = {
                "philosopher": name,
                "model": label,
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "responses": responses,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            done = pi + 1
            if done % 10 == 0 or done == len(phil_list):
                print(f"  [{done}/{len(phil_list)}] done")

        print(f"  Saved to {out_dir}/")


# ──────────────────────────────────────────────
# Step 2: NORMALIZE — parse raw responses -> 277x100 matrix
# ──────────────────────────────────────────────


def score_option(option_str):
    """Parse 'Attitude: option_value' -> (attitude_score, stem, option_value)."""
    option_str = option_str.strip()
    # Try "Attitude: value" pattern
    for attitude in sorted(ATTITUDE_SCORES.keys(), key=len, reverse=True):
        prefix = attitude + ":"
        if option_str.lower().startswith(prefix):
            value = option_str[len(prefix) :].strip()
            return ATTITUDE_SCORES[attitude], value.lower()
    # Special non-scoreable options
    for skip in [
        "accept a combination",
        "accept an alternative",
        "too unclear",
        "no fact of the matter",
        "agnostic/undecided",
        "agnostic",
    ]:
        if option_str.lower().startswith(skip):
            return None, option_str.lower()
    return None, option_str.lower()


def normalize_key(key):
    return re.sub(r"\s*\([^)]*\)", "", key).strip().lower()


def extract_stem_option(key):
    norm = normalize_key(key)
    if ":" in norm:
        stem, opt = norm.rsplit(":", 1)
        return stem.strip(), opt.strip()
    return norm, norm


def determine_most_popular(human_data):
    """From human normalized data, pick the most-popular option per multi-option stem."""
    stem_option_counts = defaultdict(lambda: defaultdict(int))
    for phil in human_data:
        for key, val in phil.get("responses", {}).items():
            if val is not None:
                stem, opt = extract_stem_option(key)
                stem_option_counts[stem][opt] += 1
    most_popular = {}
    for stem, opts in stem_option_counts.items():
        if stem not in BINARY_PAIRS:
            most_popular[stem] = max(opts, key=opts.get)
    return most_popular


def normalize_raw_to_100q(raw_dir, label):
    """Convert raw per-philosopher JSON -> single normalized JSON with 100 question scores."""
    with open(HUMAN_NORMALIZED) as f:
        human_data = json.load(f)
    most_popular = determine_most_popular(human_data)

    # Get canonical 100 question keys from human data
    canonical_keys = sorted(human_data[0]["responses"].keys())

    raw_files = sorted(Path(raw_dir).glob("*.json"))
    results = []

    n_total = 0
    n_valid = 0
    n_failed = 0

    for raw_file in raw_files:
        with open(raw_file) as f:
            data = json.load(f)

        phil_name = data["philosopher"]
        raw_responses = data.get("responses", {})

        # Collect all scored options for this philosopher: stem -> {option -> score}
        stem_scores = defaultdict(dict)

        for q_key, resp_data in raw_responses.items():
            parsed = resp_data.get("parsed", [])
            q_stem = normalize_key(q_key).lower()
            n_total += 1

            for option_str in parsed:
                score, option_value = score_option(option_str)
                if score is not None:
                    # Extract which option this is for
                    stem_scores[q_stem][option_value] = score
                    n_valid += 1

            if not parsed or all(score_option(o)[0] is None for o in parsed):
                n_failed += 1

        # Build normalized 100-question response dict
        norm_responses = {}
        for canon_key in canonical_keys:
            stem, target_opt = extract_stem_option(canon_key)

            if stem in BINARY_PAIRS:
                pos_opt, neg_opt = BINARY_PAIRS[stem]
                if pos_opt in stem_scores.get(stem, {}):
                    norm_responses[canon_key] = stem_scores[stem][pos_opt]
                elif neg_opt in stem_scores.get(stem, {}):
                    # Complement recovery
                    norm_responses[canon_key] = 1.0 - stem_scores[stem][neg_opt]
                else:
                    norm_responses[canon_key] = None
            else:
                # Multi-option: use most popular option among humans
                if target_opt in stem_scores.get(stem, {}):
                    norm_responses[canon_key] = stem_scores[stem][target_opt]
                else:
                    norm_responses[canon_key] = None

        results.append(
            {
                "name": phil_name,
                "responses": norm_responses,
            }
        )

    return results, n_total, n_valid, n_failed


def cmd_normalize(args):
    """Parse raw responses and produce normalized 100-question JSONs."""
    models = resolve_models(args.models)
    prompt_variant = getattr(args, "prompt", "paper")
    dir_suffix = f"_{prompt_variant}" if prompt_variant != "paper" else ""
    os.makedirs(NORM_DIR, exist_ok=True)

    for label, _ in models:
        raw_dir = os.path.join(RAW_DIR, f"{label}{dir_suffix}")
        if not os.path.isdir(raw_dir):
            print(f"No raw data for '{label}' at {raw_dir}. Run 'collect' first.")
            continue

        data, n_total, n_valid, n_failed = normalize_raw_to_100q(raw_dir, label)
        out_file = os.path.join(NORM_DIR, f"{label}{dir_suffix}_normalized.json")
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)

        # Stats
        n_phil = len(data)
        n_responses = sum(
            1 for p in data for v in p["responses"].values() if v is not None
        )
        n_possible = n_phil * 100
        resp_rate = n_responses / n_possible * 100 if n_possible else 0
        variances = []
        for j in range(100):
            key = sorted(data[0]["responses"].keys())[j]
            vals = [
                p["responses"][key] for p in data if p["responses"][key] is not None
            ]
            if len(vals) > 1:
                variances.append(np.var(vals))
        mean_var = np.mean(variances) if variances else 0

        print(f"\n{label}:")
        print(f"  Philosophers: {n_phil}")
        print(f"  Response rate: {resp_rate:.1f}% ({n_responses}/{n_possible})")
        print(f"  Mean per-Q variance: {mean_var:.4f}")
        print(f"  Parse failures: {n_failed} / {n_total}")
        print(f"  Saved: {out_file}")


# ──────────────────────────────────────────────
# Step 3: ANALYZE — compute all paper metrics
# ──────────────────────────────────────────────


def load_matrix(filepath):
    """Load normalized JSON -> (277 x 100 numpy matrix, phil_names, question_keys)."""
    with open(filepath) as f:
        data = json.load(f)
    phil_names = [p["name"] for p in data]
    question_keys = sorted(data[0]["responses"].keys())
    mat = np.full((len(phil_names), len(question_keys)), np.nan)
    for i, p in enumerate(data):
        for j, qk in enumerate(question_keys):
            v = p["responses"].get(qk)
            if v is not None:
                mat[i, j] = v
    return mat, phil_names, question_keys


def kl_divergence(p, q, n_bins=N_BINS, value_range=(0, 1)):
    """KL(P || Q) with histogram binning and Laplace smoothing."""
    bins = np.linspace(value_range[0], value_range[1], n_bins + 1)
    p_hist = np.histogram(p, bins=bins, density=False)[0].astype(float) + 1
    q_hist = np.histogram(q, bins=bins, density=False)[0].astype(float) + 1
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def js_divergence(p, q, n_bins=N_BINS, value_range=(0, 1)):
    """JS divergence with histogram binning."""
    bins = np.linspace(value_range[0], value_range[1], n_bins + 1)
    p_hist = np.histogram(p, bins=bins, density=False)[0].astype(float) + 1
    q_hist = np.histogram(q, bins=bins, density=False)[0].astype(float) + 1
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(jensenshannon(p_hist, q_hist) ** 2)


def pairwise_corr_matrix(mat):
    """Compute question-question correlation matrix using pairwise deletion."""
    n_q = mat.shape[1]
    corr = np.full((n_q, n_q), np.nan)
    for i in range(n_q):
        for j in range(i, n_q):
            mask = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            if mask.sum() > 5:
                r, _ = pearsonr(mat[mask, i], mat[mask, j])
                corr[i, j] = corr[j, i] = r
            if i == j:
                corr[i, j] = 1.0
    return corr


def mantel_test(corr1, corr2, n_perms=MANTEL_PERMS):
    """Mantel test: correlation between upper-triangular elements of two matrices."""
    n = corr1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = corr1[idx]
    v2 = corr2[idx]
    mask = ~np.isnan(v1) & ~np.isnan(v2)
    v1, v2 = v1[mask], v2[mask]
    if len(v1) < 10:
        return np.nan, 1.0
    r_obs, _ = pearsonr(v1, v2)
    n_greater = 0
    for _ in range(n_perms):
        perm = np.random.permutation(len(v2))
        r_perm, _ = pearsonr(v1, v2[perm])
        if r_perm >= r_obs:
            n_greater += 1
    p = (n_greater + 1) / (n_perms + 1)
    return float(r_obs), float(p)


def rv_coefficient(corr1, corr2):
    """RV coefficient between two correlation matrices."""
    n = corr1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = corr1[idx]
    v2 = corr2[idx]
    mask = ~np.isnan(v1) & ~np.isnan(v2)
    v1, v2 = v1[mask], v2[mask]
    if len(v1) < 10:
        return np.nan
    # RV = trace(X'Y Y'X) / sqrt(trace(X'X)^2 * trace(Y'Y)^2)
    # For vectors: RV = (v1 . v2)^2 / (|v1|^2 * |v2|^2)
    # But proper RV is on the matrices; approximate with correlation of upper tri
    r, _ = pearsonr(v1, v2)
    return float(r**2)


def cmd_analyze(args):
    """Compute paper metrics comparing Fireworks model results to human data."""
    if not os.path.exists(HUMAN_NORMALIZED):
        print(f"Human data not found: {HUMAN_NORMALIZED}")
        sys.exit(1)

    human_mat, phil_names, question_keys = load_matrix(HUMAN_NORMALIZED)
    human_corr = pairwise_corr_matrix(human_mat)

    # Flatten human values for KL/JS
    human_flat = human_mat[~np.isnan(human_mat)]

    # Human stats
    human_vars = [
        np.nanvar(human_mat[:, j])
        for j in range(human_mat.shape[1])
        if np.sum(~np.isnan(human_mat[:, j])) > 1
    ]
    print(f"\nHuman baseline:")
    print(f"  Mean per-Q variance: {np.mean(human_vars):.4f}")
    print(
        f"  Response rate: {np.sum(~np.isnan(human_mat))}/{human_mat.size} "
        f"({np.sum(~np.isnan(human_mat))/human_mat.size*100:.1f}%)"
    )

    # Find all normalized files
    models = resolve_models(args.models) if hasattr(args, "models") else []
    norm_files = {}
    for label, _ in models:
        path = os.path.join(NORM_DIR, f"{label}_normalized.json")
        if os.path.exists(path):
            norm_files[label] = path

    if not norm_files:
        # Auto-discover
        for f in Path(NORM_DIR).glob("*_normalized.json"):
            label = f.stem.replace("_normalized", "")
            if label != "human_survey":
                norm_files[label] = str(f)

    if not norm_files:
        print(f"No normalized data found in {NORM_DIR}/. Run 'normalize' first.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("TABLE: Overall Response Matrix Similarity (paper Table 4)")
    print(f"{'='*80}")
    print(
        f"{'Model':<20} {'KL Div':>8} {'JS Div':>8} {'Pearson r':>10} {'Per-Q Var':>10} {'Resp%':>8}"
    )
    print("-" * 70)

    model_corrs = {}
    for label, path in sorted(norm_files.items()):
        mat, _, _ = load_matrix(path)

        # Response rate
        n_resp = np.sum(~np.isnan(mat))
        resp_pct = n_resp / mat.size * 100

        # Per-question variance
        variances = [
            np.nanvar(mat[:, j])
            for j in range(mat.shape[1])
            if np.sum(~np.isnan(mat[:, j])) > 1
        ]
        mean_var = np.mean(variances) if variances else 0

        # Flatten for KL/JS
        model_flat = mat[~np.isnan(mat)]
        kl = kl_divergence(human_flat, model_flat)
        js = js_divergence(human_flat, model_flat)

        # Pearson on overlapping non-NaN entries
        mask = ~np.isnan(human_mat) & ~np.isnan(mat)
        if mask.sum() > 10:
            r, _ = pearsonr(human_mat[mask], mat[mask])
        else:
            r = np.nan

        print(
            f"{label:<20} {kl:>8.3f} {js:>8.3f} {r:>10.3f} {mean_var:>10.4f} {resp_pct:>7.1f}%"
        )

        # Store correlation matrix for Mantel/RV
        model_corrs[label] = pairwise_corr_matrix(mat)

    # Question correlation structure
    print(f"\n{'='*80}")
    print("TABLE: Question Correlation Structure (paper Table 7)")
    print(f"{'='*80}")
    print(
        f"{'Model':<20} {'Mantel r':>10} {'Mantel p':>10} {'RV coeff':>10} {'Corr KL':>8} {'Corr JS':>8}"
    )
    print("-" * 70)

    human_corr_flat = human_corr[np.triu_indices(human_corr.shape[0], k=1)]
    human_corr_valid = human_corr_flat[~np.isnan(human_corr_flat)]

    for label, mcorr in sorted(model_corrs.items()):
        r_mantel, p_mantel = mantel_test(human_corr, mcorr)
        rv = rv_coefficient(human_corr, mcorr)

        # KL/JS on correlation distributions
        model_corr_flat = mcorr[np.triu_indices(mcorr.shape[0], k=1)]
        model_corr_valid = model_corr_flat[~np.isnan(model_corr_flat)]
        corr_kl = kl_divergence(human_corr_valid, model_corr_valid, value_range=(-1, 1))
        corr_js = js_divergence(human_corr_valid, model_corr_valid, value_range=(-1, 1))

        sig = "**" if p_mantel < 0.01 else ("*" if p_mantel < 0.05 else "")
        print(
            f"{label:<20} {r_mantel:>9.3f}{sig:<1} {p_mantel:>10.3f} {rv:>10.3f} {corr_kl:>8.3f} {corr_js:>8.3f}"
        )

    # PCA comparison
    print(f"\n{'='*80}")
    print("PCA: Variance explained by top-6 components")
    print(f"{'='*80}")

    def run_pca(mat, n_components=6):
        """Simple PCA with mean imputation."""
        m = mat.copy()
        col_means = np.nanmean(m, axis=0)
        for j in range(m.shape[1]):
            mask = np.isnan(m[:, j])
            m[mask, j] = col_means[j]
        m -= m.mean(axis=0)
        _, s, _ = np.linalg.svd(m, full_matrices=False)
        var_explained = (s**2) / (s**2).sum()
        return var_explained[:n_components]

    human_pca = run_pca(human_mat)
    print(
        f"{'Human':<20} top-6: {sum(human_pca)*100:.1f}%  "
        f"({', '.join(f'{v*100:.1f}%' for v in human_pca)})"
    )

    for label, path in sorted(norm_files.items()):
        mat, _, _ = load_matrix(path)
        pca = run_pca(mat)
        print(
            f"{label:<20} top-6: {sum(pca)*100:.1f}%  "
            f"({', '.join(f'{v*100:.1f}%' for v in pca)})"
        )

    print(f"\n{'='*80}")
    print("Heterogeneity collapse ratio (Human var / Model var)")
    print(f"{'='*80}")
    human_mean_var = np.mean(human_vars)
    for label, path in sorted(norm_files.items()):
        mat, _, _ = load_matrix(path)
        variances = [
            np.nanvar(mat[:, j])
            for j in range(mat.shape[1])
            if np.sum(~np.isnan(mat[:, j])) > 1
        ]
        mv = np.mean(variances) if variances else 0.001
        ratio = human_mean_var / mv if mv > 0 else float("inf")
        print(f"  {label:<20} {ratio:.1f}x lower variance")


# ──────────────────────────────────────────────
# DPO pipeline commands (convert/upload/train/deploy)
# ──────────────────────────────────────────────


def _parse_prompt_to_messages(prompt):
    """Parse 'System: ...\n\nUser: ...' into chat messages list."""
    messages = []
    if prompt.startswith("System: "):
        parts = prompt.split("\n\nUser: ", 1)
        messages.append(
            {"role": "system", "content": parts[0].removeprefix("System: ")}
        )
        if len(parts) > 1:
            messages.append({"role": "user", "content": parts[1]})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _create_and_upload_dataset(dataset_id, filepath):
    """Create dataset entry + upload file to Fireworks."""
    if not API_KEY or not ACCOUNT_ID:
        print("Set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID.")
        sys.exit(1)
    # Count lines for exampleCount
    with open(filepath) as f:
        n_examples = sum(1 for _ in f)
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/datasets"
    resp = requests.post(
        url,
        headers=headers("application/json"),
        json={
            "datasetId": dataset_id,
            "dataset": {"userUploaded": {}, "exampleCount": str(n_examples)},
        },
    )
    if resp.status_code == 200:
        print(f"Dataset '{dataset_id}' created ({n_examples} examples).")
    elif resp.status_code == 409:
        print(f"Dataset '{dataset_id}' exists, reusing.")
    else:
        print(f"Error creating dataset: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    upload_url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/datasets/{dataset_id}:upload"
    with open(filepath, "rb") as f:
        resp = requests.post(
            upload_url,
            headers=headers(),
            files={"file": (os.path.basename(filepath), f)},
        )
    if "already uploaded" in resp.text:
        print(f"Dataset already uploaded, reusing.")
    else:
        resp.raise_for_status()
        print(f"Uploaded {filepath}.")


def _poll_job(endpoint):
    """Poll a fine-tuning job endpoint until terminal state."""
    terminal = {
        "JOB_STATE_COMPLETED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_EARLY_STOPPED",
    }
    print("Polling (every 60s)...")
    while True:
        r = requests.get(endpoint, headers=headers())
        r.raise_for_status()
        state = r.json().get("state", "UNKNOWN")
        print(f"  {datetime.now().strftime('%H:%M:%S')} - {state}")
        if state in terminal:
            if state != "JOB_STATE_COMPLETED":
                msg = r.json().get("status", {}).get("message")
                if msg:
                    print(f"  {msg}")
            break
        time.sleep(60)


# ── DPO commands ─────────────────────────────


def cmd_convert(args):
    """Convert local DPO data to Fireworks DPO format (preferred/non_preferred)."""
    count = 0
    with open(DPO_INPUT_JSONL) as fin, open(DPO_CONVERTED_JSONL, "w") as fout:
        for line in fin:
            row = json.loads(line)
            messages = _parse_prompt_to_messages(row["prompt"])
            fout.write(
                json.dumps(
                    {
                        "input": {"messages": messages},
                        "preferred_output": [
                            {"role": "assistant", "content": row["chosen"]}
                        ],
                        "non_preferred_output": [
                            {"role": "assistant", "content": row["rejected"]}
                        ],
                    }
                )
                + "\n"
            )
            count += 1
    print(f"DPO: Converted {count} examples -> {DPO_CONVERTED_JSONL}")


def cmd_upload(args):
    _create_and_upload_dataset(DPO_DATASET_ID, DPO_CONVERTED_JSONL)


def cmd_train(args):
    if not API_KEY or not ACCOUNT_ID:
        print("Set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID.")
        sys.exit(1)
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/dpoJobs"
    tc = dict(DPO_TRAINING_CONFIG)
    tc["outputModel"] = f"accounts/{ACCOUNT_ID}/models/{DPO_OUTPUT_MODEL_ID}"
    payload = {
        "dataset": f"accounts/{ACCOUNT_ID}/datasets/{DPO_DATASET_ID}",
        "baseModel": FT_BASE_MODEL,
        "trainingConfig": tc,
        "lossConfig": DPO_LOSS_CONFIG,
    }
    resp = requests.post(
        url,
        headers=headers("application/json"),
        json=payload,
        params={"dpoJobId": DPO_JOB_ID},
    )
    resp.raise_for_status()
    job = resp.json()
    job_name = job.get("name", "unknown")
    job_id = job_name.split("/")[-1]
    print(f"DPO job created: {job_name}")
    _poll_job(f"{BASE_URL}/accounts/{ACCOUNT_ID}/dpoJobs/{job_id}")


# ── SFT commands ─────────────────────────────


def cmd_sft_convert(args):
    """Convert local DPO data to Fireworks SFT format (uses only the 'chosen' response)."""
    count = 0
    with open(DPO_INPUT_JSONL) as fin, open(SFT_CONVERTED_JSONL, "w") as fout:
        for line in fin:
            row = json.loads(line)
            messages = _parse_prompt_to_messages(row["prompt"])
            # SFT format: system + user + assistant (the chosen/correct response)
            messages.append({"role": "assistant", "content": row["chosen"]})
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    print(f"SFT: Converted {count} examples -> {SFT_CONVERTED_JSONL}")


def cmd_sft_upload(args):
    _create_and_upload_dataset(SFT_DATASET_ID, SFT_CONVERTED_JSONL)


def cmd_sft_train(args):
    if not API_KEY or not ACCOUNT_ID:
        print("Set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID.")
        sys.exit(1)
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/supervisedFineTuningJobs"
    payload = {
        "dataset": f"accounts/{ACCOUNT_ID}/datasets/{SFT_DATASET_ID}",
        "baseModel": FT_BASE_MODEL,
        "outputModel": f"accounts/{ACCOUNT_ID}/models/{SFT_OUTPUT_MODEL_ID}",
        "epochs": SFT_TRAINING_CONFIG["epochs"],
        "learningRate": SFT_TRAINING_CONFIG["learningRate"],
        "loraRank": SFT_TRAINING_CONFIG["loraRank"],
    }
    resp = requests.post(
        url,
        headers=headers("application/json"),
        json=payload,
        params={"supervisedFineTuningJobId": SFT_JOB_ID},
    )
    if resp.status_code != 200:
        print(f"SFT job error: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    job = resp.json()
    job_name = job.get("name", "unknown")
    job_id = job_name.split("/")[-1]
    print(f"SFT job created: {job_name}")
    _poll_job(f"{BASE_URL}/accounts/{ACCOUNT_ID}/supervisedFineTuningJobs/{job_id}")


# ── Deploy (shared) ──────────────────────────


def cmd_deploy(args):
    if not API_KEY or not ACCOUNT_ID:
        print("Set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID.")
        sys.exit(1)
    model = args.model or f"accounts/{ACCOUNT_ID}/models/{DPO_OUTPUT_MODEL_ID}"
    resp = requests.post(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/deployments",
        headers=headers("application/json"),
        json={"baseModel": model},
    )
    resp.raise_for_status()
    print(f"Deployment created: {resp.json().get('name')}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fireworks AI: End-to-end philosopher survey pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    model_help = (
        f"Comma-separated. Available: {','.join(BASELINE_MODELS.keys())},"
        "llama3p18b-dpo,llama3p18b-sft"
    )

    prompt_help = (
        "Prompt variant: 'paper' (original) or 'v2' (simplified). Default: paper"
    )

    p = sub.add_parser(
        "collect", help="Query models for all philosopher-question pairs"
    )
    p.add_argument("--models", type=str, default=None, help=model_help)
    p.add_argument("--prompt", type=str, default="paper", help=prompt_help)

    p = sub.add_parser(
        "normalize", help="Parse raw responses -> 100q normalized format"
    )
    p.add_argument("--models", type=str, default=None, help=model_help)
    p.add_argument("--prompt", type=str, default="paper", help=prompt_help)

    p = sub.add_parser("analyze", help="Compute all paper metrics vs human data")
    p.add_argument("--models", type=str, default=None, help=model_help)

    # DPO pipeline
    sub.add_parser("convert", help="Convert DPO data to Fireworks format")
    sub.add_parser("upload", help="Upload DPO dataset to Fireworks")
    sub.add_parser("train", help="Launch DPO fine-tuning job and poll")

    # SFT pipeline
    sub.add_parser("sft-convert", help="Convert DPO data to SFT format (chosen only)")
    sub.add_parser("sft-upload", help="Upload SFT dataset to Fireworks")
    sub.add_parser("sft-train", help="Launch SFT fine-tuning job and poll")

    # Shared
    p = sub.add_parser("deploy", help="Deploy a fine-tuned model")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model resource name (defaults to DPO model)",
    )

    args = parser.parse_args()
    cmds = {
        "collect": cmd_collect,
        "normalize": cmd_normalize,
        "analyze": cmd_analyze,
        "convert": cmd_convert,
        "upload": cmd_upload,
        "train": cmd_train,
        "sft-convert": cmd_sft_convert,
        "sft-upload": cmd_sft_upload,
        "sft-train": cmd_sft_train,
        "deploy": cmd_deploy,
    }
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
