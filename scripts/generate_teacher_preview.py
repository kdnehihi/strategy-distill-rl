import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import exact_match
from src.teacher.local_teacher import generate_teacher_output, load_local_teacher
from src.teacher.formatting import (
    build_canonical_teacher_output,
    check_teacher_format,
    extract_teacher_fields,
    is_valid_teacher_format,
)
from src.utils.prompts import build_strategy_teacher_prompt


def read_jsonl(path: Path, limit: int):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_teacher_record(example, teacher_output: str):
    fields = extract_teacher_fields(teacher_output)
    canonical_output = build_canonical_teacher_output(fields)

    strategy = fields["strategy"]
    teacher_answer = fields["answer"]
    is_correct = exact_match(teacher_answer, example["ground_truth"])
    format_checks = check_teacher_format(teacher_output)
    is_format_valid = is_valid_teacher_format(teacher_output)

    return {
        "id": example["id"],
        "question": example["question"],
        "ground_truth": example["ground_truth"],
        "raw_teacher_output": teacher_output,
        "teacher_output": canonical_output,
        "strategy": strategy,
        "reasoning": fields["reasoning"],
        "teacher_answer": teacher_answer,
        "is_correct": is_correct,
        "is_format_valid": int(is_format_valid),
        "is_usable": int(is_correct and is_format_valid),
        "format_checks": format_checks,
    }


def build_retry_prompt(question: str) -> str:
    return (
        build_strategy_teacher_prompt(question)
        + "\n\nRemember: the only parsed part is the <final> block. Your output "
        "must contain <strategy>, <reasoning>, and <answer> inside <final>, and "
        "must end immediately after </final>. Keep <reasoning> concise, at most "
        "3 sentences."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a small teacher preview file for cleaned GSM8K."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face model name or local model path for the teacher.",
    )
    parser.add_argument(
        "--input-path",
        default="data/gsm8k_clean_train.jsonl",
        help="Cleaned GSM8K JSONL input path.",
    )
    parser.add_argument(
        "--output-path",
        default="data/gsm8k_teacher_preview.jsonl",
        help="Teacher preview JSONL output path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of examples to run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0.0 for deterministic generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry generation when the output does not match the required format.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    examples = read_jsonl(input_path, limit=args.num_samples)
    tokenizer, model = load_local_teacher(args.model_name)

    teacher_records = []
    for example in tqdm(examples, desc="Generating teacher traces"):
        prompt = build_strategy_teacher_prompt(example["question"])
        record = None

        for attempt in range(args.max_retries + 1):
            teacher_output = generate_teacher_output(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            record = build_teacher_record(example, teacher_output)

            if record["is_format_valid"]:
                break

            prompt = build_retry_prompt(example["question"])

        teacher_records.append(record)

    write_jsonl(teacher_records, output_path)

    correct = sum(record["is_correct"] for record in teacher_records)
    usable = sum(record["is_usable"] for record in teacher_records)
    total = len(teacher_records)
    print(f"Saved {total} teacher records to {output_path}")
    print(f"Exact match: {correct}/{total}")
    print(f"Usable records: {usable}/{total}")

    print("\nSample teacher records:")
    for record in teacher_records[:3]:
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
