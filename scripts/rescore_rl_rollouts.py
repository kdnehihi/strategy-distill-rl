import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.student.evaluation import build_student_eval_record
from src.student.rewards import score_student_output


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


def rescore_group(group: dict) -> dict:
    example = {
        "id": group["id"],
        "question": group["question"],
        "ground_truth": group["ground_truth"],
    }

    rescored_outputs = []
    for output in group.get("outputs", []):
        raw_output = output.get("raw_output") or output.get("model_output") or ""
        eval_record = build_student_eval_record(example, raw_output)
        reward, reward_parts = score_student_output(example, eval_record)

        updated_output = {
            **output,
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
            "reward": reward,
            "reward_parts": reward_parts,
        }
        rescored_outputs.append(updated_output)

    rewards = [output["reward"] for output in rescored_outputs]
    return {
        **group,
        "outputs": rescored_outputs,
        "correct_count": sum(output["is_correct"] for output in rescored_outputs),
        "format_valid_count": sum(
            output["is_format_valid"] for output in rescored_outputs
        ),
        "has_reward_variance": len(set(rewards)) > 1,
        "reward_version": "correctness_format_target_quantity_v1",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute RL rollout rewards without regenerating model outputs."
    )
    parser.add_argument(
        "--input-path",
        default="data/rl_rollouts_1p5b_sft_g8.jsonl",
        help="Existing grouped rollout JSONL.",
    )
    parser.add_argument(
        "--output-path",
        default="data/rl_rollouts_1p5b_sft_g8_target_reward.jsonl",
        help="Output JSONL with updated reward and reward_parts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    groups = read_jsonl(input_path)
    rescored_groups = [rescore_group(group) for group in groups]
    write_jsonl(rescored_groups, output_path)

    total_outputs = sum(len(group.get("outputs", [])) for group in rescored_groups)
    useful_groups = sum(group.get("has_reward_variance", False) for group in rescored_groups)
    correct_outputs = sum(
        output["is_correct"]
        for group in rescored_groups
        for output in group.get("outputs", [])
    )
    format_valid_outputs = sum(
        output["is_format_valid"]
        for group in rescored_groups
        for output in group.get("outputs", [])
    )

    print(f"Saved {len(rescored_groups)} rescored rollout groups to {output_path}")
    print(f"Total outputs: {total_outputs}")
    if total_outputs:
        print(f"Correct outputs: {correct_outputs}/{total_outputs}")
        print(f"Format-valid outputs: {format_valid_outputs}/{total_outputs}")
    print(f"Groups with reward variance: {useful_groups}/{len(rescored_groups)}")


if __name__ == "__main__":
    main()
