import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import exact_match
from src.teacher.formatting import check_teacher_format, extract_teacher_fields
from src.utils.prompts import STRATEGIES


REQUIRED_FIELDS = {
    "id": int,
    "question": str,
    "ground_truth": str,
    "raw_teacher_output": str,
    "teacher_output": str,
    "strategy": str,
    "reasoning": str,
    "teacher_answer": str,
    "is_correct": int,
    "is_format_valid": int,
    "is_usable": int,
    "format_checks": dict,
}


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def validate_schema(record):
    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"missing field: {field}")
            continue
        if not isinstance(record[field], expected_type):
            actual = type(record[field]).__name__
            errors.append(f"{field} should be {expected_type.__name__}, got {actual}")
    return errors


def validate_record(record):
    errors = []
    errors.extend(validate_schema(record))

    if errors:
        return errors

    format_checks = check_teacher_format(record["teacher_output"])
    if not all(format_checks.values()):
        failed = [name for name, passed in format_checks.items() if not passed]
        errors.append(f"format failed: {failed}")

    fields = extract_teacher_fields(record["teacher_output"])
    if fields["strategy"] != record["strategy"]:
        errors.append("strategy does not match parsed teacher_output")
    if fields["reasoning"] != record["reasoning"]:
        errors.append("reasoning does not match parsed teacher_output")
    if fields["answer"] != record["teacher_answer"]:
        errors.append("teacher_answer does not match parsed teacher_output")

    recomputed_correct = exact_match(record["teacher_answer"], record["ground_truth"])
    if recomputed_correct != record["is_correct"]:
        errors.append("is_correct does not match recomputed exact match")

    recomputed_format_valid = int(all(format_checks.values()))
    if recomputed_format_valid != record["is_format_valid"]:
        errors.append("is_format_valid does not match recomputed format checks")

    recomputed_usable = int(recomputed_correct and recomputed_format_valid)
    if recomputed_usable != record["is_usable"]:
        errors.append("is_usable does not match correctness and format validity")

    return errors


def print_strategy_distribution(records):
    counts = Counter(record.get("strategy") for record in records)
    total = len(records)

    print("\nStrategy distribution:")
    for strategy in STRATEGIES:
        count = counts.get(strategy, 0)
        percent = count / total if total else 0
        print(f"  {strategy}: {count} ({percent:.1%})")

    unknown_count = sum(
        count for strategy, count in counts.items() if strategy not in STRATEGIES
    )
    if unknown_count:
        print(f"  unknown_or_invalid: {unknown_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a usable teacher dataset and report strategy counts."
    )
    parser.add_argument(
        "--path",
        default="data/gsm8k_teacher_preview_5000.jsonl",
        help="Teacher JSONL dataset path.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum detailed record errors to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    records = read_jsonl(path)

    ids = [record.get("id") for record in records]
    duplicate_ids = sorted(
        record_id for record_id, count in Counter(ids).items() if count > 1
    )

    total = len(records)
    correct = sum(record.get("is_correct", 0) for record in records)
    format_valid = sum(record.get("is_format_valid", 0) for record in records)
    usable = sum(record.get("is_usable", 0) for record in records)

    print(f"Validating: {path}")
    print(f"Records: {total}")
    print(f"Unique ids: {len(set(ids))}")
    print(f"Duplicate ids: {len(duplicate_ids)}")
    if duplicate_ids:
        print(f"First duplicate ids: {duplicate_ids[:args.max_errors]}")

    if total:
        print(f"is_correct: {correct}/{total} ({correct / total:.1%})")
        print(f"is_format_valid: {format_valid}/{total} ({format_valid / total:.1%})")
        print(f"is_usable: {usable}/{total} ({usable / total:.1%})")

    record_errors = []
    for record in records:
        errors = validate_record(record)
        if errors:
            record_errors.append((record.get("id"), errors))

    print(f"\nInvalid records: {len(record_errors)}")
    for record_id, errors in record_errors[: args.max_errors]:
        print(f"  id={record_id}: {errors}")

    print_strategy_distribution(records)

    if duplicate_ids or record_errors:
        print("\nValidation result: FAILED")
    else:
        print("\nValidation result: PASSED")


if __name__ == "__main__":
    main()
