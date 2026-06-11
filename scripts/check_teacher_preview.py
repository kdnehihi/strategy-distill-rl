import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.teacher.formatting import check_teacher_format


def parse_args():
    parser = argparse.ArgumentParser(description="Check teacher preview format.")
    parser.add_argument(
        "--path",
        default="data/gsm8k_teacher_preview.jsonl",
        help="Teacher preview JSONL path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    totals = {
        "has_final_block": 0,
        "has_exactly_one_final_block": 0,
        "has_strategy_tag": 0,
        "strategy_is_allowed": 0,
        "has_reasoning_tag": 0,
        "has_answer_tag": 0,
        "answer_only_number": 0,
        "reasoning_non_empty": 0,
        "reasoning_has_no_xml_tags": 0,
        "no_text_after_final": 0,
    }
    failed_ids = {key: [] for key in totals}

    for record in records:
        output = record.get("raw_teacher_output")
        checks = check_teacher_format(output)
        for key in totals:
            if checks[key]:
                totals[key] += 1
            else:
                failed_ids[key].append(record["id"])

    total = len(records)
    print(f"Checked {total} records from {path}")
    for key, passed in totals.items():
        failed = total - passed
        print(f"{key}: {passed}/{total} passed, {failed} failed")
        if failed:
            print(f"  failed ids: {failed_ids[key][:20]}")


if __name__ == "__main__":
    main()
