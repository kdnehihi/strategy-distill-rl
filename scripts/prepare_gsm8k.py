import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_gsm8k import clean_split
from src.data.load_gsm8k import load_gsm8k


def read_sample_records(path, n=3):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for _, line in zip(range(n), f):
            records.append(json.loads(line))
    return records


def main():
    dataset = load_gsm8k()

    train_path = Path("data/gsm8k_clean_train.jsonl")
    test_path = Path("data/gsm8k_clean_test.jsonl")

    train_count = clean_split(dataset["train"], train_path)
    test_count = clean_split(dataset["test"], test_path)

    print(f"Saved {train_count} train examples to {train_path}")
    print(f"Saved {test_count} test examples to {test_path}")

    print("\nSample train records:")
    for record in read_sample_records(train_path):
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
