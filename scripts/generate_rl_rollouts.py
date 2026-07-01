import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_student import load_student_model
from scripts.train_sft_lora import ensure_torchao_compatibility
from src.student.evaluation import build_student_eval_record
from src.teacher.local_teacher import (
    generate_teacher_outputs_hf,
    generate_teacher_outputs_vllm,
    load_vllm_teacher,
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


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def score_rollout(eval_record):
    """Simple RLVR reward for rollout debugging.

    Keep this intentionally small and interpretable before implementing DAPO:
    correctness dominates, with a small format bonus/penalty.
    """
    if eval_record["is_correct"]:
        reward = 1.0
    else:
        reward = 0.0

    if eval_record["is_format_valid"]:
        reward += 0.1
    else:
        reward -= 0.2

    return round(reward, 4)


def build_rollout_record(example, raw_outputs):
    prompt = build_strategy_student_prompt(example["question"])
    outputs = []

    for generation_index, raw_output in enumerate(raw_outputs):
        eval_record = build_student_eval_record(example, raw_output)
        outputs.append(
            {
                "generation_index": generation_index,
                "raw_output": raw_output,
                "model_output": eval_record["model_output"],
                "strategy": eval_record["strategy"],
                "reasoning": eval_record["reasoning"],
                "model_answer": eval_record["model_answer"],
                "loose_model_answer": eval_record["loose_model_answer"],
                "is_correct": eval_record["is_correct"],
                "is_loose_correct": eval_record["is_loose_correct"],
                "is_format_valid": eval_record["is_format_valid"],
                "is_usable": eval_record["is_usable"],
                "format_checks": eval_record["format_checks"],
                "reward": score_rollout(eval_record),
            }
        )

    rewards = [output["reward"] for output in outputs]
    correct_count = sum(output["is_correct"] for output in outputs)
    format_valid_count = sum(output["is_format_valid"] for output in outputs)

    return {
        "id": example["id"],
        "question": example["question"],
        "ground_truth": example["ground_truth"],
        "prompt": prompt,
        "num_generations": len(outputs),
        "correct_count": correct_count,
        "format_valid_count": format_valid_count,
        "has_reward_variance": len(set(rewards)) > 1,
        "outputs": outputs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple sampled student outputs per prompt for RLVR."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base student model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default="checkpoints/student_sft/balanced_r16_a32_4000",
        help="Optional PEFT LoRA adapter path.",
    )
    parser.add_argument(
        "--input-path",
        default="data/gsm8k_clean_train.jsonl",
        help="Cleaned GSM8K JSONL input file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/rl_rollouts.jsonl",
        help="Output JSONL path for grouped rollouts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of prompts to sample. Use -1 for all prompts.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of sampled outputs per prompt.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
        help="Generation backend.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.num_samples == -1 else args.num_samples
    examples = read_jsonl(Path(args.input_path), limit=limit)

    if args.backend == "vllm" and args.adapter_path:
        raise ValueError(
            "This simple vLLM rollout script expects a merged model or no adapter. "
            "Use --backend hf for PEFT adapter rollouts."
        )

    if args.backend == "vllm":
        tokenizer, llm = load_vllm_teacher(
            args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        model = None
    else:
        ensure_torchao_compatibility()
        tokenizer, model = load_student_model(args.model_name, args.adapter_path)
        llm = None

    rollout_records = []
    for batch_examples in tqdm(
        list(batched(examples, args.batch_size)),
        desc="Generating RL rollouts",
    ):
        prompts = [
            build_strategy_student_prompt(example["question"])
            for example in batch_examples
        ]

        if args.backend == "vllm":
            grouped_outputs = generate_teacher_outputs_vllm(
                tokenizer=tokenizer,
                llm=llm,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n=args.num_generations,
            )
            if args.num_generations == 1:
                grouped_outputs = [[output] for output in grouped_outputs]
        else:
            flat_outputs = generate_teacher_outputs_hf(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_generations,
            )
            grouped_outputs = [
                flat_outputs[
                    start : start + args.num_generations
                ]
                for start in range(0, len(flat_outputs), args.num_generations)
            ]

        for example, raw_outputs in zip(batch_examples, grouped_outputs):
            rollout_records.append(build_rollout_record(example, raw_outputs))

        write_jsonl(rollout_records, Path(args.output_path))

    total_outputs = sum(record["num_generations"] for record in rollout_records)
    correct_outputs = sum(
        output["is_correct"]
        for record in rollout_records
        for output in record["outputs"]
    )
    format_valid_outputs = sum(
        output["is_format_valid"]
        for record in rollout_records
        for output in record["outputs"]
    )
    useful_groups = sum(record["has_reward_variance"] for record in rollout_records)

    print(f"Saved {len(rollout_records)} rollout groups to {args.output_path}")
    print(f"Total outputs: {total_outputs}")
    if total_outputs:
        print(f"Correct outputs: {correct_outputs}/{total_outputs}")
        print(f"Format-valid outputs: {format_valid_outputs}/{total_outputs}")
    print(f"Groups with reward variance: {useful_groups}/{len(rollout_records)}")


if __name__ == "__main__":
    main()
