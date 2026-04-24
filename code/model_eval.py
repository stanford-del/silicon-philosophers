"""
LLM Philosopher Response Generator - Colab/Notebook Version

This version is optimized for Google Colab and Jupyter notebooks:
- Single process with GPU support
- Easy model switching
- Progress tracking with tqdm
- Simplified configuration
- MODIFIED: Supports loading fine-tuned LoRA models
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time
import json
import os
from datetime import datetime
from tqdm.auto import tqdm
import re

# ============================================================================
# CONFIGURATION - Modify these variables to customize your run
# ============================================================================

# Export HuggingFace token from Databricks secret or environment variable.
# Set HF_TOKEN in your environment before running this script.
import os

access_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not access_token:
    raise RuntimeError(
        "HuggingFace token not set. Export HF_TOKEN=<your-token> before running."
    )

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import os
from datetime import datetime
from tqdm.auto import tqdm
import re

# ============================================================================
# CONFIGURATION - Modify these variables to customize your run
# ============================================================================

# Model Configuration
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Change to any HuggingFace model
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # <1B parameter model from HuggingFace
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # <1B parameter model from HuggingFace

# Fine-tuned Model Configuration (NEW!)
USE_FINETUNED_MODEL = True  # Set to True to use your fine-tuned DPO model
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Base model used for fine-tuning
LORA_ADAPTER_PATH = (
    "./llama_philosopher_dpo/final_model"  # Path to fine-tuned LoRA adapter
)

# Other examples:
# - "google/gemma-2-2b-it"
# - "meta-llama/Llama-3.2-1B-Instruct"
# - "mistralai/Mistral-7B-Instruct-v0.2"

# Generation Settings
MAX_NEW_TOKENS = 100
MAX_RETRIES = 5  # Number of retry attempts for invalid responses
TEMPERATURE = 0.0  # Set to 0 for deterministic responses
DO_SAMPLE = False  # Set to True for sampling, False for greedy decoding

# File Paths
DATA_DIR = "."  # Directory containing input JSON files
OUTPUT_DIR = f"llm_responses_{MODEL_NAME.split('/')[-1]}"
RESUME_FILE = f"llm_responses_progress_{MODEL_NAME.split('/')[-1]}.json"

# Input files (modify if your files have different names)
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
QUESTIONS_FILE = "question_answer_options.json"

# Testing Configuration
TEST_LIMIT = None  # Set to a number (e.g., 10) for testing, or None for full run

# Device Configuration (auto-detected)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Other examples:
# - "google/gemma-2-2b-it"
# - "meta-llama/Llama-3.2-1B-Instruct"
# - "mistralai/Mistral-7B-Instruct-v0.2"

# Generation Settings
MAX_NEW_TOKENS = 100
MAX_RETRIES = 5  # Number of retry attempts for invalid responses
TEMPERATURE = 0.0  # Set to 0 for deterministic responses
DO_SAMPLE = False  # Set to True for sampling, False for greedy decoding

# File Paths
DATA_DIR = "."  # Directory containing input JSON files
OUTPUT_DIR = f"llm_responses_{MODEL_NAME.split('/')[-1]}"
RESUME_FILE = f"llm_responses_progress_{MODEL_NAME.split('/')[-1]}.json"

# Input files (modify if your files have different names)
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
QUESTIONS_FILE = "question_answer_options.json"

# Testing Configuration
TEST_LIMIT = None  # Set to a number (e.g., 10) for testing, or None for full run

# Device Configuration (auto-detected)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def normalize_option(option):
    """Normalize an option for comparison (handle formatting variations)"""
    if not option:
        return ""

    # Convert to string and lowercase
    normalized = str(option).lower().strip()

    # Remove extra spaces
    normalized = " ".join(normalized.split())

    # Remove trailing punctuation (., !, etc.)
    normalized = normalized.rstrip(".,!?;")

    # Normalize colons:
    # - Remove multiple consecutive colons
    while "::" in normalized:
        normalized = normalized.replace("::", ":")

    # - Normalize space around colons (ensure single space after colon, none before)
    normalized = normalized.replace(" :", ":")  # Remove space before colon
    normalized = normalized.replace(":", ": ")  # Add space after colon
    normalized = normalized.replace(":  ", ": ")  # Remove double spaces after colon

    # Final cleanup of extra spaces
    normalized = " ".join(normalized.split())

    return normalized


def parse_response_list(response_text):
    """Parse LLM response to extract list of options"""
    # Clean up the response
    response_text = response_text.strip()

    # Try to find JSON list in the response
    # Look for [...] pattern
    json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)

    if json_match:
        try:
            # Try to parse as JSON
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                # Clean up each item
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass

    # If JSON parsing failed, try to extract items manually
    # Look for quoted strings
    quoted_items = re.findall(r'"([^"]+)"', response_text)
    if quoted_items:
        return quoted_items

    # Look for single-quoted strings
    single_quoted = re.findall(r"'([^']+)'", response_text)
    if single_quoted:
        return single_quoted

    # If still nothing, treat the whole response as a single item
    return [response_text]


def validate_response(parsed_response, valid_options):
    """Validate that all response items are in the valid options list (with fuzzy matching)"""
    if not parsed_response:
        return False, "Empty response - no options selected"

    if not isinstance(parsed_response, list):
        return False, f"Response is not a list: {type(parsed_response)}"

    # Create multiple normalized lookups of valid options
    # Map 1: Standard normalization
    normalized_valid = {normalize_option(opt): opt for opt in valid_options}

    # Map 2: Without any colons (to match "Accept a combination" with "Accept: a combination")
    normalized_no_colon = {
        normalize_option(opt).replace(":", ""): opt for opt in valid_options
    }

    invalid_items = []
    matched_items = []

    for item in parsed_response:
        normalized_item = normalize_option(item)
        normalized_item_no_colon = normalized_item.replace(":", "")

        # Try exact match first
        if item in valid_options:
            matched_items.append(item)
            continue

        # Try normalized match
        if normalized_item in normalized_valid:
            matched_items.append(normalized_valid[normalized_item])
            continue

        # Try match without colons (handles "Accept: a combination" matching "Accept a combination")
        if normalized_item_no_colon in normalized_no_colon:
            matched_items.append(normalized_no_colon[normalized_item_no_colon])
            continue

        # No match found
        invalid_items.append(item)

    if invalid_items:
        # Show what was invalid and a sample of valid options
        sample_valid = valid_options[:3]
        return (
            False,
            f"Invalid options: {invalid_items}. Valid options include: {sample_valid}...",
        )

    # Return success with normalized matched items
    return True, "Valid"


def build_persona_prompt(persona):
    """Build persona prompt (second-person, no name)"""
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

    # PhD information
    if phd or phd_country or phd_year:
        prompt += "Your Educational Background:\n"
        if phd:
            prompt += f"- PhD from {phd}"
            if phd_country:
                prompt += f" ({phd_country})"
            if phd_year:
                prompt += f" in {phd_year}"
            prompt += "\n"
        elif phd_country and phd_year:
            prompt += f"- PhD from {phd_country} in {phd_year}\n"
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


def build_question_prompt(question_key, answer_options):
    """Build question prompt with options"""
    prompt = """You are answering a survey by the reputable website PhilPapers, which collects
responses across various philosophical domains based on your background. You may select one or multiple options if you deem it appropriate to accurately represent your position.

Please respond with your chosen option(s) as a JSON list. If selecting multiple options, ensure they are logically consistent.

Examples of valid responses:
- ["Accept: physicalism"]
- ["Accept: physicalism", "Reject: non-physicalism"]
- ["Lean towards: physicalism", "Lean against: non-physicalism"]

Given your philosophical profile above, please express your stance on the following question.

Question: """

    prompt += question_key + "\n\n"
    prompt += "Available response options:\n"

    for option in answer_options:
        prompt += f"- {option}\n"

    prompt += """

Respond ONLY with the JSON list, no other text or explanation.

"""

    return prompt


# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================


class PhilosopherResponseGenerator:
    """Main class for generating philosopher responses"""

    def __init__(self, model_name=MODEL_NAME, device=DEVICE, data_dir=DATA_DIR):
        self.model_name = model_name
        self.device = device
        self.data_dir = data_dir
        self.model = None
        self.tokenizer = None
        self.philosophers = None
        self.question_options = None

    def load_model(self):
        """Load model and tokenizer"""
        if USE_FINETUNED_MODEL:
            print(f"Loading FINE-TUNED model")
            print(f"Base model: {BASE_MODEL_NAME}")
            print(f"LoRA adapter: {LORA_ADAPTER_PATH}")
            print(f"Device: {self.device}")

            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME, token=access_token
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            if self.device == "cuda":
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map=None,  # Don't use device_map with LoRA
                    low_cpu_mem_usage=True,
                    token=access_token,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME, device_map="cpu", token=access_token
                )

            # Load LoRA adapter
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            self.model.eval()

            # Move to device if CUDA
            if self.device == "cuda":
                self.model = self.model.cuda()

            print("✓ Fine-tuned model loaded successfully")

        else:
            print(f"Loading BASE model: {self.model_name}")
            print(f"Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=access_token
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate device mapping
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use fp16 for faster GPU inference
                    device_map="auto",
                    token=access_token,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="cpu", token=access_token
                )

            print("✓ Model loaded successfully")

    def load_data(self):
        """Load philosopher and question data"""
        print("\nLoading data files...")

        phil_path = os.path.join(self.data_dir, PHILOSOPHERS_FILE)
        quest_path = os.path.join(self.data_dir, QUESTIONS_FILE)

        with open(phil_path) as f:
            self.philosophers = json.load(f)

        with open(quest_path) as f:
            self.question_options = json.load(f)

        print(f"✓ Loaded {len(self.philosophers)} philosophers")
        print(f"✓ Loaded {len(self.question_options)} questions")

    def generate_response(
        self, philosopher, question_key, answer_options, max_retries=MAX_RETRIES
    ):
        """Generate a single response for a philosopher-question pair with validation and retry"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build full prompt once
        persona_prompt = build_persona_prompt(philosopher)
        question_prompt = build_question_prompt(question_key, answer_options)
        full_prompt = persona_prompt + question_prompt

        # Prepare for model
        messages = [{"role": "user", "content": full_prompt}]

        # Apply chat template (format depends on model)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if chat template not available
            text = full_prompt

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        attempts = []
        total_time = 0

        for attempt in range(max_retries):
            try:
                # Generate
                start = time.time()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE if DO_SAMPLE else None,
                        do_sample=DO_SAMPLE,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                elapsed = time.time() - start
                total_time += elapsed

                # Extract response
                generated_tokens = outputs[0][input_length:]
                raw_response = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                # Parse response
                parsed_response = parse_response_list(raw_response)

                # Validate response
                is_valid, validation_msg = validate_response(
                    parsed_response, answer_options
                )

                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "parsed": parsed_response,
                        "raw": raw_response,
                        "valid": is_valid,
                        "validation_msg": validation_msg,
                        "time": elapsed,
                    }
                )

                if is_valid:
                    return {
                        "success": True,
                        "parsed": parsed_response,
                        "raw": raw_response,
                        "generation_time": total_time,
                        "attempts": attempt + 1,
                        "all_attempts": attempts,
                    }

                # If invalid and not last attempt, continue to retry
                if attempt < max_retries - 1:
                    continue
                else:
                    # Max retries reached with invalid response
                    return {
                        "success": False,
                        "error": f"Max retries reached. Last validation: {validation_msg}",
                        "parsed": parsed_response,
                        "raw": raw_response,
                        "generation_time": total_time,
                        "attempts": max_retries,
                        "all_attempts": attempts,
                    }

            except Exception as e:
                import traceback

                error_detail = f"{type(e).__name__}: {str(e)}"
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "error": error_detail,
                        "traceback": traceback.format_exc(),
                        "time": 0,
                    }
                )

                if attempt < max_retries - 1:
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Exception after {max_retries} attempts: {error_detail}",
                        "generation_time": total_time,
                        "attempts": max_retries,
                        "all_attempts": attempts,
                    }

    def generate_all_responses(
        self,
        output_dir=OUTPUT_DIR,
        resume_file=RESUME_FILE,
        test_limit=TEST_LIMIT,
        batch_size=10,
    ):
        """Generate responses for all philosopher-question combinations"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load progress
        completed = set()
        if os.path.exists(resume_file):
            with open(resume_file, "r") as f:
                progress_data = json.load(f)
                completed = set(progress_data.get("completed", []))

        # Get all questions
        all_questions = list(self.question_options.keys())

        # Calculate total work
        total_combinations = len(self.philosophers) * len(all_questions)

        print(f"\n{'='*80}")
        print("CONFIGURATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Max retries: {MAX_RETRIES}")
        print(f"Output directory: {output_dir}")
        if test_limit:
            print(f"⚠️  TEST MODE: Limited to {test_limit} items")
        print(f"Total combinations: {total_combinations}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {total_combinations - len(completed)}")
        print(f"{'='*80}\n")

        # Build list of tasks to process
        tasks = []

        for philosopher in self.philosophers:
            phil_name = philosopher.get("name", "Unknown")

            for question_key in all_questions:
                # Check if we've reached test limit
                if test_limit is not None and len(tasks) >= test_limit:
                    break

                # Create unique ID for this combination
                combo_id = f"{phil_name}||{question_key}"

                # Skip if already completed
                if combo_id in completed:
                    continue

                # Add task
                answer_options = self.question_options[question_key]
                tasks.append(
                    {
                        "philosopher": philosopher,
                        "question_key": question_key,
                        "answer_options": answer_options,
                        "combo_id": combo_id,
                        "phil_name": phil_name,
                    }
                )

            if test_limit is not None and len(tasks) >= test_limit:
                break

        print(f"Processing {len(tasks)} tasks...\n")

        # Process tasks with progress bar
        results_batch = []
        start_time = time.time()
        retry_count = 0
        failed_count = 0
        failed_items = []

        pbar = tqdm(tasks, desc="Processing", unit="item")

        for task in pbar:
            phil_name = task["phil_name"]
            question_key = task["question_key"]
            combo_id = task["combo_id"]

            # Generate response
            result = self.generate_response(
                task["philosopher"], task["question_key"], task["answer_options"]
            )

            # Update progress bar
            if result["success"]:
                status = "✓"
                if result.get("attempts", 1) > 1:
                    retry_count += 1
            else:
                status = "✗"
                failed_count += 1
                failed_items.append(
                    {
                        "philosopher": phil_name,
                        "question": question_key,
                        "error": result.get("error", "Unknown error"),
                    }
                )

            pbar.set_postfix(
                {"status": status, "retries": retry_count, "failed": failed_count}
            )

            # Store result
            full_result = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "philosopher": {
                    "name": phil_name,
                    "areas_of_specialization": task["philosopher"].get(
                        "areas_of_specialization", []
                    ),
                    "areas_of_interest": task["philosopher"].get(
                        "areas_of_interest", []
                    ),
                    "phd_institution": task["philosopher"].get("phd_institution"),
                    "phd_country": task["philosopher"].get("phd_country"),
                    "year_of_phd_degree": task["philosopher"].get("year_of_phd_degree"),
                    "current_institution": task["philosopher"].get(
                        "current_institution"
                    ),
                    "current_institution_country": task["philosopher"].get(
                        "current_institution_country"
                    ),
                },
                "question": question_key,
                "response": {
                    "parsed": result.get("parsed", []),
                    "raw": result.get("raw", ""),
                    "success": result["success"],
                    "error": result.get("error"),
                    "generation_time": result["generation_time"],
                    "attempts": result.get("attempts", 1),
                    "all_attempts": result.get("all_attempts", []),
                },
            }

            results_batch.append(full_result)
            completed.add(combo_id)

            # Save batch periodically
            if len(results_batch) >= batch_size:
                batch_filename = os.path.join(
                    output_dir, f"batch_{int(time.time())}.json"
                )
                with open(batch_filename, "w") as f:
                    json.dump(results_batch, f, indent=2)
                results_batch = []

                # Save progress
                with open(resume_file, "w") as f:
                    json.dump({"completed": list(completed)}, f)

        # Save remaining results
        if results_batch:
            batch_filename = os.path.join(
                output_dir, f"batch_final_{int(time.time())}.json"
            )
            with open(batch_filename, "w") as f:
                json.dump(results_batch, f, indent=2)

        # Save progress
        with open(resume_file, "w") as f:
            json.dump({"completed": list(completed)}, f)

        # Summary
        total_time = time.time() - start_time
        processed = len(tasks)
        success_count = processed - failed_count

        print(f"\n{'='*80}")
        print("COMPLETE!")
        print(f"{'='*80}")
        print(f"Total processed: {processed}")
        print(
            f"Successful: {success_count} ({success_count/processed*100:.1f}%)"
            if processed > 0
            else "Successful: 0"
        )
        print(
            f"Required retry: {retry_count} ({retry_count/processed*100:.1f}%)"
            if processed > 0
            else "Required retry: 0"
        )
        print(
            f"Failed: {failed_count} ({failed_count/processed*100:.1f}%)"
            if processed > 0
            else "Failed: 0"
        )
        print(f"Total time: {total_time/60:.1f} minutes")
        print(
            f"Average: {total_time/processed:.2f} seconds per item"
            if processed > 0
            else "N/A"
        )
        print(f"Results saved in: {output_dir}/")

        # Show failed items if any
        if failed_items:
            print(f"\n{'='*80}")
            print(f"FAILED ITEMS ({len(failed_items)} total):")
            print(f"{'='*80}")
            for idx, item in enumerate(failed_items[:10], 1):  # Show first 10
                print(f"{idx}. {item['philosopher'][:30]}")
                print(f"   Question: {item['question'][:50]}")
                print(f"   Error: {item['error'][:70]}")
            if len(failed_items) > 10:
                print(f"\n... and {len(failed_items) - 10} more failed items")

        print(f"{'='*80}")

        return {
            "total": processed,
            "success": success_count,
            "failed": failed_count,
            "retries": retry_count,
            "time": total_time,
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for script execution"""
    # Create generator
    generator = PhilosopherResponseGenerator(
        model_name=MODEL_NAME, device=DEVICE, data_dir=DATA_DIR
    )

    # Load model and data
    generator.load_model()
    generator.load_data()

    # Generate responses
    results = generator.generate_all_responses(
        output_dir=OUTPUT_DIR, resume_file=RESUME_FILE, test_limit=TEST_LIMIT
    )

    return results


if __name__ == "__main__":
    main()
