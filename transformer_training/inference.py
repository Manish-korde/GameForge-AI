"""
inference.py

Interactive command-line tool to try out the fine-tuned GameForge
Transformer model. Loads ../models/gameforge_t5/ and lets you type in
a game concept request, then prints the structured JSON output.

Run:
    python inference.py

Then type a prompt when asked, e.g.:
    Create a cyberpunk game where a bounty hunter explores a neon city.

Type "quit" or "exit" to stop.
"""

import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as e:
    print(f"ERROR: Missing package: {e}")
    print("Activate .venv_transformer and install requirements_transformer.txt first.")
    sys.exit(1)

TASK_PREFIX = "game concept: "
MAX_NEW_TOKENS = 512

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models" / "gameforge_t5"

REQUIRED_TOP_LEVEL_FIELDS = {
    "theme", "environment", "characters", "weapons", "props", "visualStyle",
}


def load_model():
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        print("Run train_transformer.py first to produce a trained model.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GameForge Transformer from {MODEL_DIR} (device: {device})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR)).to(device)
    except Exception as e:
        print(f"ERROR loading model/tokenizer: {e}")
        sys.exit(1)

    model.eval()
    print("Model loaded.\n")
    return tokenizer, model, device


def strip_code_fences(text: str) -> str:
    """Remove accidental ```json ... ``` or ``` ... ``` fences the model
    might occasionally emit."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence (and optional 'json' tag)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


def generate_game_spec(prompt_text, tokenizer, model, device):
    input_text = TASK_PREFIX + prompt_text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned_output = strip_code_fences(raw_output)
    return raw_output, cleaned_output


def validate_schema(obj):
    if not isinstance(obj, dict):
        return False, "Top-level output is not a JSON object."
    actual_fields = set(obj.keys())
    if actual_fields != REQUIRED_TOP_LEVEL_FIELDS:
        missing = REQUIRED_TOP_LEVEL_FIELDS - actual_fields
        extra = actual_fields - REQUIRED_TOP_LEVEL_FIELDS
        msg_parts = []
        if missing:
            msg_parts.append(f"missing fields: {sorted(missing)}")
        if extra:
            msg_parts.append(f"unexpected fields: {sorted(extra)}")
        return False, "; ".join(msg_parts)
    return True, "OK"


def main():
    tokenizer, model, device = load_model()

    print("GameForge AI - Transformer Inference")
    print("Type a natural-language game concept request, or 'quit' to exit.\n")

    while True:
        try:
            prompt_text = input("Enter game concept: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt_text:
            continue
        if prompt_text.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        raw_output, cleaned_output = generate_game_spec(prompt_text, tokenizer, model, device)

        try:
            parsed = json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            print("\n--- JSON PARSE FAILED ---")
            print(f"Error: {e}")
            print("Raw model output (for debugging):")
            print(raw_output)
            print("-------------------------\n")
            continue

        is_valid, message = validate_schema(parsed)

        print("\n--- Generated Game Specification ---")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print("-------------------------------------")
        if is_valid:
            print("Schema check: PASSED\n")
        else:
            print(f"Schema check: FAILED ({message})\n")


if __name__ == "__main__":
    main()
