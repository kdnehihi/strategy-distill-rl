import json
from pathlib import Path

from tqdm import tqdm

from src.utils.parsers import extract_gsm8k_answer


def clean_split(split, output_path):
    """Write a GSM8K split to JSONL with parsed final answers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(tqdm(split, desc=f"Writing {output_path.name}")):
            raw_answer = example["answer"]
            ground_truth = extract_gsm8k_answer(raw_answer)

            record = {
                "id": idx,
                "question": example["question"],
                "ground_truth": ground_truth,
                "raw_answer": raw_answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    return saved
