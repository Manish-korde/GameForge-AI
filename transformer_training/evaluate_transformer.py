"""
evaluate_transformer.py

Evaluates the fine-tuned GameForge AI FLAN-T5 model on test.jsonl.

The test set is NOT used during training.

For every test example this script:

1. Loads the trained model.
2. Generates a prediction.
3. Attempts to parse the prediction as JSON.
4. Checks the required GameForge schema.
5. Compares scalar fields.
6. Compares characters/weapons/props by name.
7. Saves every prediction for manual inspection.

Outputs:

    evaluation_results.json
    evaluation_summary.json

inside:

    models/gameforge_t5/

Run from project root:

    python transformer_training\evaluate_transformer.py
"""

import json
import math
import sys
from pathlib import Path

import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gameforge_t5"
)

TEST_FILE = (
    Path(__file__).resolve().parent
    / "test.jsonl"
)

RESULTS_FILE = (
    MODEL_DIR
    / "evaluation_results.json"
)

SUMMARY_FILE = (
    MODEL_DIR
    / "evaluation_summary.json"
)

TASK_PREFIX = "game concept: "

MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 512

NUM_BEAMS = 4


REQUIRED_TOP_LEVEL_FIELDS = {
    "theme",
    "environment",
    "characters",
    "weapons",
    "props",
    "visualStyle",
}

LIST_FIELDS = [
    "characters",
    "weapons",
    "props",
]

SCALAR_FIELDS = [
    "theme",
    "environment",
    "visualStyle",
]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():

    if not MODEL_DIR.exists():

        print(
            f"ERROR: Model directory not found:\n"
            f"  {MODEL_DIR}"
        )

        print(
            "\nRun train_transformer.py first."
        )

        sys.exit(1)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("LOADING GAMEFORGE AI TRANSFORMER")
    print("=" * 70)

    print(f"Model directory : {MODEL_DIR}")
    print(f"Device          : {device}")

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR)
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(MODEL_DIR)
        )

        model = model.to(device)

        model.eval()

    except Exception as e:

        print(
            "\nERROR loading trained model:"
        )

        print(e)

        sys.exit(1)

    print("\nModel loaded successfully.")

    return tokenizer, model, device


# ============================================================================
# TEST DATA
# ============================================================================

def load_test_set():

    if not TEST_FILE.exists():

        print(
            f"ERROR: Test file not found:\n"
            f"  {TEST_FILE}"
        )

        sys.exit(1)

    examples = []

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                examples.append(record)

            except json.JSONDecodeError:

                print(
                    f"WARNING: malformed JSON "
                    f"on test line {line_number}"
                )

    return examples


# ============================================================================
# JSON CLEANUP
# ============================================================================

def strip_code_fences(text):

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        # Remove opening fence.
        if lines:
            lines = lines[1:]

        # Remove closing fence.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================================
# PREDICTION
# ============================================================================

def generate_prediction(
    prompt_text,
    tokenizer,
    model,
    device,
):

    input_text = (
        TASK_PREFIX
        + prompt_text
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,

            num_beams=NUM_BEAMS,

            do_sample=False,

            max_new_tokens=MAX_NEW_TOKENS,

            early_stopping=True,
        )

    decoded = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return strip_code_fences(decoded)


# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def normalize_name_set(items):

    names = set()

    if not isinstance(items, list):

        return names

    for item in items:

        if not isinstance(item, dict):
            continue

        name = item.get("name")

        if isinstance(name, str):

            name = (
                name
                .strip()
                .lower()
            )

            if name:
                names.add(name)

    return names


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(obj):

    if not isinstance(obj, dict):
        return False

    # Top-level keys must match exactly.
    if set(obj.keys()) != REQUIRED_TOP_LEVEL_FIELDS:
        return False

    # Scalar fields.
    for field in SCALAR_FIELDS:

        if not isinstance(
            obj.get(field),
            str,
        ):
            return False

    # List fields.
    for field in LIST_FIELDS:

        value = obj.get(field)

        if not isinstance(value, list):
            return False

        for item in value:

            if not isinstance(item, dict):
                return False

            if not isinstance(
                item.get("name"),
                str,
            ):
                return False

            if not isinstance(
                item.get("desc"),
                str,
            ):
                return False

    return True


# ============================================================================
# JACCARD
# ============================================================================

def jaccard_score(
    predicted,
    target,
):

    predicted_names = normalize_name_set(
        predicted
    )

    target_names = normalize_name_set(
        target
    )

    if (
        not predicted_names
        and not target_names
    ):
        return 1.0

    union = (
        predicted_names
        | target_names
    )

    intersection = (
        predicted_names
        & target_names
    )

    if not union:
        return 1.0

    return (
        len(intersection)
        / len(union)
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate():

    tokenizer, model, device = load_model()

    test_examples = load_test_set()

    total = len(test_examples)

    if total == 0:

        print(
            "\nNo test examples found."
        )

        return

    print("\n" + "=" * 70)
    print("GAMEFORGE AI TRANSFORMER EVALUATION")
    print("=" * 70)

    print(
        f"Test examples: {total}"
    )

    print()

    # ------------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------------

    valid_json_count = 0

    schema_valid_count = 0

    exact_match_count = 0

    scalar_correct = {
        field: 0
        for field in SCALAR_FIELDS
    }

    scalar_total = {
        field: 0
        for field in SCALAR_FIELDS
    }

    list_scores = {
        field: []
        for field in LIST_FIELDS
    }

    prediction_results = []

    # ------------------------------------------------------------------------
    # Process test examples
    # ------------------------------------------------------------------------

    for i, example in enumerate(
        test_examples,
        start=1,
    ):

        prompt = example.get(
            "input",
            "",
        )

        target_text = example.get(
            "target",
            "",
        )

        # ------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------

        try:

            target_obj = json.loads(
                target_text
            )

        except json.JSONDecodeError:

            print(
                f"[{i}/{total}] "
                f"GROUND TRUTH INVALID"
            )

            continue

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        try:

            prediction_text = generate_prediction(
                prompt,
                tokenizer,
                model,
                device,
            )

        except Exception as e:

            print(
                f"[{i}/{total}] "
                f"GENERATION ERROR: {e}"
            )

            prediction_results.append(
                {
                    "index": i,
                    "input": prompt,
                    "target": target_obj,
                    "prediction_text": "",
                    "prediction": None,
                    "valid_json": False,
                    "schema_valid": False,
                    "exact_match": False,
                    "generation_error": str(e),
                }
            )

            continue

        # ------------------------------------------------------------
        # Parse JSON
        # ------------------------------------------------------------

        prediction_obj = None

        valid_json = False

        schema_valid = False

        exact_match = False

        try:

            prediction_obj = json.loads(
                prediction_text
            )

            valid_json = True

        except json.JSONDecodeError:

            prediction_obj = None

        # ------------------------------------------------------------
        # Schema
        # ------------------------------------------------------------

        if valid_json:

            schema_valid = validate_schema(
                prediction_obj
            )

        # ------------------------------------------------------------
        # Exact match
        # ------------------------------------------------------------

        if schema_valid:

            if prediction_obj == target_obj:

                exact_match = True

                exact_match_count += 1

        # ------------------------------------------------------------
        # Counters
        # ------------------------------------------------------------

        if valid_json:

            valid_json_count += 1

        if schema_valid:

            schema_valid_count += 1

        # ------------------------------------------------------------
        # Scalar fields
        # ------------------------------------------------------------

        if schema_valid:

            for field in SCALAR_FIELDS:

                scalar_total[field] += 1

                predicted_value = (
                    str(
                        prediction_obj.get(
                            field,
                            "",
                        )
                    )
                    .strip()
                    .lower()
                )

                target_value = (
                    str(
                        target_obj.get(
                            field,
                            "",
                        )
                    )
                    .strip()
                    .lower()
                )

                if (
                    predicted_value
                    == target_value
                ):

                    scalar_correct[field] += 1

        # ------------------------------------------------------------
        # List fields
        # ------------------------------------------------------------

        if schema_valid:

            for field in LIST_FIELDS:

                score = jaccard_score(
                    prediction_obj.get(
                        field,
                        [],
                    ),
                    target_obj.get(
                        field,
                        [],
                    ),
                )

                list_scores[field].append(
                    score
                )

        # ------------------------------------------------------------
        # Status
        # ------------------------------------------------------------

        if schema_valid:

            status = "OK"

        elif valid_json:

            status = "JSON-ONLY"

        else:

            status = "INVALID"

        print(
            f"[{i:3d}/{total}] {status}"
        )

        # ------------------------------------------------------------
        # Save complete result
        # ------------------------------------------------------------

        prediction_results.append(
            {
                "index": i,

                "input": prompt,

                "target": target_obj,

                "prediction_text": prediction_text,

                "prediction": prediction_obj,

                "valid_json": valid_json,

                "schema_valid": schema_valid,

                "exact_match": exact_match,
            }
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(
        f"Test examples       : "
        f"{total}"
    )

    print(
        f"Valid JSON          : "
        f"{valid_json_count}/{total} "
        f"({100 * valid_json_count / total:.1f}%)"
    )

    print(
        f"Schema-valid        : "
        f"{schema_valid_count}/{total} "
        f"({100 * schema_valid_count / total:.1f}%)"
    )

    print(
        f"Exact match         : "
        f"{exact_match_count}/{total} "
        f"({100 * exact_match_count / total:.1f}%)"
    )

    # ------------------------------------------------------------------------
    # Scalar accuracy
    # ------------------------------------------------------------------------

    print(
        "\nScalar field accuracy "
        "(schema-valid predictions):"
    )

    scalar_accuracy = {}

    for field in SCALAR_FIELDS:

        if scalar_total[field] > 0:

            accuracy = (
                100
                * scalar_correct[field]
                / scalar_total[field]
            )

            scalar_accuracy[field] = accuracy

            print(
                f"  {field:<12}: "
                f"{accuracy:.1f}% "
                f"({scalar_correct[field]}/"
                f"{scalar_total[field]})"
            )

        else:

            scalar_accuracy[field] = None

            print(
                f"  {field:<12}: N/A"
            )

    # ------------------------------------------------------------------------
    # List similarity
    # ------------------------------------------------------------------------

    print(
        "\nList field similarity "
        "(name-set Jaccard):"
    )

    list_accuracy = {}

    for field in LIST_FIELDS:

        scores = list_scores[field]

        if scores:

            average = (
                sum(scores)
                / len(scores)
            )

            percentage = (
                100 * average
            )

            list_accuracy[field] = percentage

            print(
                f"  {field:<12}: "
                f"{percentage:.1f}% average overlap"
            )

        else:

            list_accuracy[field] = None

            print(
                f"  {field:<12}: N/A"
            )

    # =========================================================================
    # SAVE DETAILED RESULTS
    # =========================================================================

    print(
        "\nSaving detailed predictions..."
    )

    try:

        with RESULTS_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                prediction_results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved:\n  {RESULTS_FILE}"
        )

    except Exception as e:

        print(
            f"WARNING: Could not save "
            f"evaluation results: {e}"
        )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    summary = {
        "model": "google/flan-t5-small",
        "model_directory": str(MODEL_DIR),
        "test_examples": total,

        "valid_json": {
            "count": valid_json_count,
            "percentage": (
                100
                * valid_json_count
                / total
            ),
        },

        "schema_valid": {
            "count": schema_valid_count,
            "percentage": (
                100
                * schema_valid_count
                / total
            ),
        },

        "exact_match": {
            "count": exact_match_count,
            "percentage": (
                100
                * exact_match_count
                / total
            ),
        },

        "scalar_accuracy": scalar_accuracy,

        "list_name_jaccard": list_accuracy,
    }

    try:

        with SUMMARY_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        print(
            f"Saved:\n  {SUMMARY_FILE}"
        )

    except Exception as e:

        print(
            f"WARNING: Could not save "
            f"evaluation summary: {e}"
        )

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if valid_json_count == total:

        print(
            "✅ All predictions are valid JSON."
        )

    else:

        print(
            "⚠️ Some predictions are not valid JSON."
        )

    if schema_valid_count == total:

        print(
            "✅ All predictions follow the GameForge schema."
        )

    else:

        print(
            "⚠️ Some predictions do not follow the schema."
        )

    if exact_match_count > 0:

        print(
            f"Exact matches: "
            f"{exact_match_count}"
        )

    print(
        "\nDetailed predictions are available "
        "for manual inspection."
    )

    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    evaluate()