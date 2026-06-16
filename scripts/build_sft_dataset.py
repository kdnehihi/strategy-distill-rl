import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_record(record):
    return {
        "id": record["id"],
        "question": record["question"],
        "target": record["teacher_output"],
        "ground_truth": record["ground_truth"],
        "strategy": record["strategy"],
        "teacher_answer": record["teacher_answer"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SFT train/val JSONL files from usable teacher traces."
    )
    parser.add_argument(
        "--teacher-path",
        default="data/gsm8k_teacher_preview_5000.jsonl",
        help="Validated teacher JSONL path.",
    )
    parser.add_argument(
        "--train-output",
        default="data/sft_strategy_train.jsonl",
        help="Output path for SFT train records.",
    )
    parser.add_argument(
        "--val-output",
        default="data/sft_strategy_val.jsonl",
        help="Output path for SFT validation records.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=4000,
        help="Number of usable records to put in train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling records before splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = read_jsonl(Path(args.teacher_path))
    usable_records = [record for record in records if record.get("is_usable") == 1]

    rng = random.Random(args.seed)
    rng.shuffle(usable_records)

    sft_records = [build_sft_record(record) for record in usable_records]
    train_records = sft_records[: args.train_size]
    val_records = sft_records[args.train_size :]

    write_jsonl(train_records, Path(args.train_output))
    write_jsonl(val_records, Path(args.val_output))

    print(f"Usable teacher records: {len(usable_records)}")
    print(f"Train records: {len(train_records)} -> {args.train_output}")
    print(f"Val records: {len(val_records)} -> {args.val_output}")

    print("\nSample SFT record:")
    if train_records:
        print(json.dumps(train_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
