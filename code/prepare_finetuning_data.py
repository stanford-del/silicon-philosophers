#!/usr/bin/env python3
"""
Prepare training data for fine-tuning from actual philosopher responses
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# Question mapping - maps response keywords to full questions
QUESTION_KEYWORDS = {
    "free will": {
        "keywords": ["free will", "compatibilism", "libertarianism", "determinism"],
        "question": "Free will: compatibilism, libertarianism, or no free will?",
        "description": "Do we have free will, and if so, what kind?",
    },
    "god": {
        "keywords": ["god", "theism", "atheism", "agnosticism"],
        "question": "God: theism or atheism?",
        "description": "Does God exist?",
    },
    "zombies": {
        "keywords": ["zombie", "zombies", "conceivable", "metaphysically possible"],
        "question": "Philosophical zombies: conceivable but not metaphysically possible, metaphysically possible, or inconceivable?",
        "description": "Are philosophical zombies possible?",
    },
    "mind": {
        "keywords": ["physicalism", "dualism", "mind-body"],
        "question": "Mind: physicalism or non-physicalism?",
        "description": "What is the relationship between mind and body?",
    },
    "trolley": {
        "keywords": ["trolley", "footbridge", "switch"],
        "question": "Trolley problem (switch): permissible or impermissible?",
        "description": "Is it permissible to switch the trolley to save five people?",
    },
    "meta-ethics": {
        "keywords": ["moral realism", "moral anti-realism", "meta-ethics"],
        "question": "Meta-ethics: moral realism or moral anti-realism?",
        "description": "Are there objective moral truths?",
    },
    "abstract objects": {
        "keywords": ["platonism", "nominalism", "abstract objects"],
        "question": "Abstract objects: Platonism or nominalism?",
        "description": "Do abstract objects exist?",
    },
    "a priori": {
        "keywords": ["a priori", "apriori", "a-priori"],
        "question": "A priori knowledge: yes or no?",
        "description": "Is there substantive a priori knowledge?",
    },
    "normative ethics": {
        "keywords": ["deontology", "consequentialism", "virtue ethics"],
        "question": "Normative ethics: deontology, consequentialism, or virtue ethics?",
        "description": "What is the correct normative ethical theory?",
    },
    "personal identity": {
        "keywords": ["personal identity", "biological view", "psychological view"],
        "question": "Personal identity: biological view, psychological view, or further-fact view?",
        "description": "What makes a person at one time the same person as at another time?",
    },
    "external world": {
        "keywords": [
            "external world",
            "skepticism",
            "non-skeptical realism",
            "idealism",
        ],
        "question": "External world: non-skeptical realism, skepticism, or idealism?",
        "description": "What should we believe about the external world?",
    },
    "aesthetic value": {
        "keywords": ["aesthetic", "beauty", "objective", "subjective"],
        "question": "Aesthetic value: objective or subjective?",
        "description": "Are aesthetic judgments objective or subjective?",
    },
}


def extract_name_from_url(url: str) -> str:
    """Extract philosopher name from profile URL"""
    slug = url.split("/profiles/")[-1] if "/profiles/" in url else "Unknown"
    return slug.replace("-", " ").title()


def match_response_to_question(response_text: str) -> Dict:
    """Match a response to its question using keyword matching"""
    response_lower = response_text.lower()

    # Score each question by keyword matches
    scores = {}
    for q_key, q_data in QUESTION_KEYWORDS.items():
        score = sum(1 for keyword in q_data["keywords"] if keyword in response_lower)
        if score > 0:
            scores[q_key] = score

    if not scores:
        return None

    # Return best match
    best_match = max(scores.items(), key=lambda x: x[1])[0]
    return QUESTION_KEYWORDS[best_match]


def create_persona_description(
    name: str, detail: Dict, verbosity: str = "detailed"
) -> str:
    """Create persona description with varying levels of detail"""

    specs = detail.get("areas_of_specialization", [])
    interests = detail.get("areas_of_interest", [])
    phd = detail.get("phd_institution")
    institution = detail.get("current_institution")

    if verbosity == "minimal":
        return f"You are {name}, a professional philosopher."

    elif verbosity == "brief":
        persona = f"You are {name}, a professional philosopher"
        if specs:
            persona += f" specializing in {', '.join(specs[:2])}"
        persona += "."
        return persona

    else:  # detailed
        persona = f"You are {name}, a professional philosopher"
        if institution:
            persona += f" at {institution}"
        if phd:
            persona += f". You earned your PhD from {phd}"
        persona += "."

        if specs:
            persona += "\n\n**Areas of Specialization:**"
            for spec in specs[:5]:
                persona += f"\n  • {spec}"

        if interests and interests != specs:
            persona += "\n\n**Areas of Interest:**"
            for interest in interests[:5]:
                persona += f"\n  • {interest}"

        return persona


def generate_explanation(
    response_text: str, specs: List[str], question_key: str
) -> str:
    """Generate a plausible explanation based on specialization"""

    explanations = {
        "free will": {
            "Ethics": "My work in ethics informs my view on moral responsibility and agency.",
            "Philosophy of Mind": "My background in philosophy of mind shapes my understanding of the relationship between consciousness and free will.",
            "Metaphysics": "My metaphysical commitments inform my position on determinism and causation.",
        },
        "god": {
            "Philosophy of Religion": "My work in philosophy of religion directly addresses questions of theistic belief.",
            "Epistemology": "My epistemological framework shapes how I evaluate arguments for God's existence.",
            "Metaphysics": "My metaphysical views inform my stance on the existence of necessary beings.",
        },
        "trolley": {
            "Ethics": "My work in normative ethics provides the framework for evaluating the permissibility of this action.",
            "Applied Ethics": "My focus on applied ethics helps me analyze real-world moral dilemmas like this.",
            "Political Philosophy": "My work on justice and rights informs my view on the trolley problem.",
        },
        "mind": {
            "Philosophy of Mind": "My specialization in philosophy of mind directly addresses the mind-body problem.",
            "Cognitive Science": "My interdisciplinary background informs my view on mental states and brain states.",
            "Consciousness": "My work on consciousness shapes my position on physicalism.",
        },
        "meta-ethics": {
            "Ethics": "My work in ethics includes meta-ethical questions about the nature of moral facts.",
            "Metaethics": "My specialization directly addresses the ontology of moral properties.",
            "Philosophy of Language": "My work on language informs how I understand moral discourse.",
        },
    }

    # Find matching specialization
    for spec in specs:
        for key_area in explanations.get(question_key, {}):
            if key_area.lower() in spec.lower():
                return explanations[question_key][key_area]

    # Generic fallback
    return "My philosophical background and training inform this position."


def create_training_example(
    philosopher: Dict,
    response: Dict,
    question_data: Dict,
    verbosity: str = "detailed",
    format: str = "openai",
) -> Dict:
    """Create a single training example"""

    # Extract data
    name = extract_name_from_url(philosopher["profile_url"])
    detail = philosopher["detail"]
    response_text = response.get("raw_text", "")

    # Create persona
    persona = create_persona_description(name, detail, verbosity)

    # Create question prompt
    question_prompt = f"\n\nQuestion: {question_data['question']}\n\n"
    question_prompt += f"{question_data['description']}\n\n"
    question_prompt += "Provide your stance and a brief explanation."

    # Create completion
    completion = f"Stance: {response_text}\n\n"

    # Add explanation if we have specialization data
    specs = detail.get("areas_of_specialization", [])
    if specs:
        # Try to infer question key from question_data
        q_key = next(
            (
                k
                for k, v in QUESTION_KEYWORDS.items()
                if v["question"] == question_data["question"]
            ),
            None,
        )
        if q_key:
            explanation = generate_explanation(response_text, specs, q_key)
            completion += explanation

    # Format according to target
    if format == "openai":
        return {
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": question_prompt},
                {"role": "assistant", "content": completion},
            ]
        }
    else:  # huggingface
        return {
            "text": persona + question_prompt + "\n\n" + completion,
            "philosopher": name,
            "question": question_data["question"],
            "response": response_text,
        }


def prepare_training_data(
    min_demographic_fields: int = 1,
    verbosity: str = "detailed",
    format: str = "openai",
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Prepare complete training, validation, and test sets

    Args:
        min_demographic_fields: Minimum number of demographic fields required
        verbosity: Level of detail in persona ("minimal", "brief", "detailed")
        format: Output format ("openai" or "huggingface")
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """

    print("Loading data...")
    with open("philosopher_details.json") as f:
        details = json.load(f)

    with open("survey_responses_all_reprocessed.json") as f:
        responses = json.load(f)

    # Create lookup
    details_dict = {d["profile_url"]: d for d in details}

    print("Processing philosophers...")
    # Group data by philosopher
    philosopher_data = []

    for response_data in responses:
        if not response_data.get("has_survey_responses", False):
            continue

        profile_url = response_data["profile_url"]
        detail = details_dict.get(profile_url, {})

        # Check demographic requirements
        demo_count = 0
        if detail.get("areas_of_specialization"):
            demo_count += 1
        if detail.get("areas_of_interest"):
            demo_count += 1
        if detail.get("phd_institution"):
            demo_count += 1
        if detail.get("current_institution"):
            demo_count += 1

        if demo_count < min_demographic_fields:
            continue

        philosopher_data.append(
            {
                "profile_url": profile_url,
                "detail": detail,
                "responses": response_data["survey_responses"],
            }
        )

    print(
        f"Found {len(philosopher_data)} philosophers with sufficient demographic data"
    )

    # Split philosophers into train/val/test
    random.seed(random_seed)
    random.shuffle(philosopher_data)

    n = len(philosopher_data)
    test_n = int(n * 0.15)
    val_n = int(n * 0.15)

    test_philosophers = philosopher_data[:test_n]
    val_philosophers = philosopher_data[test_n : test_n + val_n]
    train_philosophers = philosopher_data[test_n + val_n :]

    print(
        f"Split: {len(train_philosophers)} train, {len(val_philosophers)} val, {len(test_philosophers)} test"
    )

    # Create training examples
    train_examples = []
    val_examples = []
    test_examples = []

    for split_name, phil_list, examples_list in [
        ("train", train_philosophers, train_examples),
        ("validation", val_philosophers, val_examples),
        ("test", test_philosophers, test_examples),
    ]:
        print(f"\nCreating {split_name} examples...")

        for philosopher in phil_list:
            for response in philosopher["responses"]:
                response_text = response.get("raw_text", "")

                if not response_text or len(response_text) < 5:
                    continue

                # Match to question
                question_data = match_response_to_question(response_text)

                if not question_data:
                    continue

                # Create example
                example = create_training_example(
                    philosopher,
                    response,
                    question_data,
                    verbosity=verbosity,
                    format=format,
                )

                examples_list.append(example)

        print(f"Created {len(examples_list)} {split_name} examples")

    return train_examples, val_examples, test_examples


def save_data(train_data, val_data, test_data, format="openai"):
    """Save training data to files"""

    if format == "openai":
        # Save as JSONL
        with open("philosopher_train.jsonl", "w") as f:
            for example in train_data:
                f.write(json.dumps(example) + "\n")

        with open("philosopher_val.jsonl", "w") as f:
            for example in val_data:
                f.write(json.dumps(example) + "\n")

        with open("philosopher_test.jsonl", "w") as f:
            for example in test_data:
                f.write(json.dumps(example) + "\n")

        print("\nSaved files:")
        print("  • philosopher_train.jsonl")
        print("  • philosopher_val.jsonl")
        print("  • philosopher_test.jsonl")

    else:  # huggingface
        with open("philosopher_train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        with open("philosopher_val.json", "w") as f:
            json.dump(val_data, f, indent=2)

        with open("philosopher_test.json", "w") as f:
            json.dump(test_data, f, indent=2)

        print("\nSaved files:")
        print("  • philosopher_train.json")
        print("  • philosopher_val.json")
        print("  • philosopher_test.json")


def main():
    """Main execution"""
    print("=" * 80)
    print("PREPARING FINE-TUNING DATA")
    print("=" * 80)

    # Prepare data
    train_data, val_data, test_data = prepare_training_data(
        min_demographic_fields=1,
        verbosity="detailed",
        format="openai",  # Change to "huggingface" for local training
        random_seed=42,
    )

    # Save data
    save_data(train_data, val_data, test_data, format="openai")

    # Print statistics
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    print(f"Training examples:   {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples:       {len(test_data)}")
    print(f"Total examples:      {len(train_data) + len(val_data) + len(test_data)}")

    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE TRAINING EXAMPLE")
    print("=" * 80)

    sample = train_data[0]
    if "messages" in sample:  # OpenAI format
        for msg in sample["messages"]:
            print(f"\n[{msg['role'].upper()}]")
            print(
                msg["content"][:300] + "..."
                if len(msg["content"]) > 300
                else msg["content"]
            )
    else:  # Huggingface format
        print(
            sample["text"][:500] + "..."
            if len(sample["text"]) > 500
            else sample["text"]
        )

    print("\n" + "=" * 80)
    print("READY FOR FINE-TUNING")
    print("=" * 80)
    print("\nNext steps:")
    print(
        "1. For OpenAI: Upload philosopher_train.jsonl to OpenAI and start fine-tuning"
    )
    print("2. For local: Use philosopher_train.json with Hugging Face Trainer")
    print("\nSee FINETUNING_STRATEGY.md for detailed instructions.")


if __name__ == "__main__":
    main()
