"""
validate_dataset.py

Validates train.jsonl, validation.jsonl, and test.jsonl BEFORE training.

Checks performed (per record):
 1. Line is valid JSON
 2. Record contains "input"
 3. Record contains "target"
 4. "target" is a string (not a nested object)
 5. "target" string can itself be parsed with json.loads()
 6. Parsed target contains EXACTLY these top-level fields:
    theme, environment, characters, weapons, props, visualStyle
 7. "characters" is a list
 8. "weapons" is a list
 9. "props" is a list
10. Each character has "name" and "desc"
11. Each weapon has "name" and "desc"
12. Each prop has "name" and "desc"
13. "input" is not empty
14. "target" is not empty
15. No duplicate "input" values within a file
16. No malformed records overall (summarized as an error count)

Run:
    python validate_dataset.py

It looks for train.jsonl / validation.jsonl / test.jsonl in the same folder
as this script. If a file is missing, it is skipped with a warning (not a
crash) so you can validate whichever files currently exist.
"""

import json
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL_FIELDS = {
    "theme",
    "environment",
    "characters",
    "weapons",
    "props",
    "visualStyle",
}

LIST_FIELDS_WITH_NAME_DESC = ["characters", "weapons", "props"]

SCRIPT_DIR = Path(__file__).resolve().parent
FILES_TO_CHECK = ["train.jsonl", "validation.jsonl", "test.jsonl"]


def validate_file(path: Path):
    """Validate a single JSONL file. Returns (summary_dict, error_list)."""
    errors = []
    seen_inputs = set()
    duplicate_inputs = 0
    valid_records = 0
    total_lines = 0

    if not path.exists():
        return None, [f"File not found: {path}"]

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue  # skip blank lines silently
            total_lines += 1

            # Check 1: valid JSON line
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                errors.append(f"[line {line_number}] Invalid JSON line: {e}")
                continue

            if not isinstance(record, dict):
                errors.append(f"[line {line_number}] Record is not a JSON object")
                continue

            # Check 2 & 3: required keys present
            if "input" not in record:
                errors.append(f"[line {line_number}] Missing 'input' key")
                continue
            if "target" not in record:
                errors.append(f"[line {line_number}] Missing 'target' key")
                continue

            input_text = record["input"]
            target_text = record["target"]

            # Check 13: input not empty
            if not isinstance(input_text, str) or not input_text.strip():
                errors.append(f"[line {line_number}] 'input' is empty or not a string")
                continue

            # Check 14: target not empty
            if not isinstance(target_text, str) or not target_text.strip():
                errors.append(f"[line {line_number}] 'target' is empty or not a string")
                continue

            # Check 4: target must be a STRING (already enforced above by isinstance)

            # Check 5: target string must parse as JSON
            try:
                target_obj = json.loads(target_text)
            except json.JSONDecodeError as e:
                errors.append(f"[line {line_number}] 'target' string is not valid JSON: {e}")
                continue

            if not isinstance(target_obj, dict):
                errors.append(f"[line {line_number}] Parsed 'target' is not a JSON object")
                continue

            # Check 6: exact top-level field set
            actual_fields = set(target_obj.keys())
            if actual_fields != REQUIRED_TOP_LEVEL_FIELDS:
                missing = REQUIRED_TOP_LEVEL_FIELDS - actual_fields
                extra = actual_fields - REQUIRED_TOP_LEVEL_FIELDS
                msg = f"[line {line_number}] Top-level fields mismatch."
                if missing:
                    msg += f" Missing: {sorted(missing)}."
                if extra:
                    msg += f" Unexpected extra: {sorted(extra)}."
                errors.append(msg)
                continue

            # Checks 7-9: list fields must be lists
            record_ok = True
            for field in LIST_FIELDS_WITH_NAME_DESC:
                if not isinstance(target_obj[field], list):
                    errors.append(f"[line {line_number}] '{field}' is not a list")
                    record_ok = False
            if not record_ok:
                continue

            # Checks 10-12: each item needs name + desc
            for field in LIST_FIELDS_WITH_NAME_DESC:
                for i, item in enumerate(target_obj[field]):
                    if not isinstance(item, dict):
                        errors.append(
                            f"[line {line_number}] '{field}[{i}]' is not an object"
                        )
                        record_ok = False
                        continue
                    if "name" not in item or "desc" not in item:
                        errors.append(
                            f"[line {line_number}] '{field}[{i}]' missing 'name' or 'desc'"
                        )
                        record_ok = False
            if not record_ok:
                continue

            # Check 15: duplicate inputs within this file
            if input_text in seen_inputs:
                duplicate_inputs += 1
                errors.append(f"[line {line_number}] Duplicate input text detected")
            else:
                seen_inputs.add(input_text)

            valid_records += 1

    summary = {
        "file": str(path.name),
        "total_lines": total_lines,
        "valid_records": valid_records,
        "invalid_records": total_lines - valid_records,
        "unique_inputs": len(seen_inputs),
        "duplicate_inputs": duplicate_inputs,
        "error_count": len(errors),
    }
    return summary, errors


def main():
    print("=" * 60)
    print("GameForge AI - Dataset Validation")
    print("=" * 60)

    overall_ok = True

    for filename in FILES_TO_CHECK:
        path = SCRIPT_DIR / filename
        print(f"\nChecking: {filename}")
        print("-" * 60)

        summary, errors = validate_file(path)

        if summary is None:
            print(f"  WARNING: {errors[0]}")
            print("  Skipping this file.")
            continue

        print(f"  Total lines read      : {summary['total_lines']}")
        print(f"  Valid records         : {summary['valid_records']}")
        print(f"  Invalid records       : {summary['invalid_records']}")
        print(f"  Unique inputs         : {summary['unique_inputs']}")
        print(f"  Duplicate inputs      : {summary['duplicate_inputs']}")
        print(f"  Errors found          : {summary['error_count']}")

        if errors:
            overall_ok = False
            print("\n  First 10 errors:")
            for err in errors[:10]:
                print(f"    - {err}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more.")
        else:
            print("  Status: PASSED (no errors found)")

    print("\n" + "=" * 60)
    if overall_ok:
        print("RESULT: All checked files passed validation.")
    else:
        print("RESULT: One or more files have validation errors.")
        print("Fix the errors above BEFORE running train_transformer.py.")
    print("=" * 60)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
