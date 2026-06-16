import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.student.evaluation import build_student_eval_record, summarize_eval_records
from src.teacher.local_teacher import (
    generate_teacher_outputs_hf,
    load_local_teacher,
)
from src.utils.prompts import build_strategy_student_prompt


def read_jsonl(path: Path, limit: int | None = None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_student_model(model_name: str, adapter_path: str | None):
    tokenizer, model = load_local_teacher(model_name)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a base or LoRA-adapted student on cleaned GSM8K."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base student model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT LoRA adapter path for SFT evaluation.",
    )
    parser.add_argument(
        "--input-path",
        default="data/gsm8k_clean_test.jsonl",
        help="Cleaned GSM8K JSONL eval file.",
    )
    parser.add_argument(
        "--output-path",
        default="runs/eval/student_eval.jsonl",
        help="JSONL path for per-example outputs.",
    )
    parser.add_argument(
        "--metrics-path",
        default="runs/eval/student_metrics.json",
        help="JSON path for aggregate metrics.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of examples to evaluate. Use -1 for all examples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to generate at once.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0.0 for deterministic evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.num_samples == -1 else args.num_samples
    examples = read_jsonl(Path(args.input_path), limit=limit)

    tokenizer, model = load_student_model(args.model_name, args.adapter_path)

    records = []
    for batch_examples in tqdm(
        list(batched(examples, args.batch_size)),
        desc="Evaluating student",
    ):
        prompts = [
            build_strategy_student_prompt(example["question"])
            for example in batch_examples
        ]
        raw_outputs = generate_teacher_outputs_hf(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        for example, raw_output in zip(batch_examples, raw_outputs):
            records.append(build_student_eval_record(example, raw_output))

        write_jsonl(records, Path(args.output_path))

    metrics = summarize_eval_records(records)
    metrics.update(
        {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "input_path": args.input_path,
        }
    )
    write_json(metrics, Path(args.metrics_path))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved outputs to {args.output_path}")
    print(f"Saved metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
