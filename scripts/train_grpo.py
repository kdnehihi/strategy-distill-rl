import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_dapo import (
    build_training_samples,
    collate_features,
    compute_token_logprobs,
    filter_dapo_groups,
    load_lora_model,
    read_jsonl,
    trainable_parameter_count,
    write_json,
)
from scripts.train_sft_lora import ensure_torchao_compatibility


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small offline GRPO-style RLVR adapter from rollout data."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base student model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default="checkpoints/student_sft/balanced_r16_a32_4000",
        help="SFT LoRA adapter used as the initial policy.",
    )
    parser.add_argument(
        "--reference-adapter-path",
        default=None,
        help=(
            "Frozen adapter used for old logprobs. Defaults to --adapter-path. "
            "Use --no-reference-model for a cheaper smoke-test mode."
        ),
    )
    parser.add_argument(
        "--no-reference-model",
        action="store_true",
        help="Compute old logprobs from the current policy before each update.",
    )
    parser.add_argument(
        "--rollout-path",
        default="data/rl_rollouts_1p5b_sft_g8.jsonl",
        help="Grouped rollout JSONL from scripts/generate_rl_rollouts.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/student_grpo/debug",
        help="Where to save the trained GRPO adapter.",
    )
    parser.add_argument("--max-groups", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def grpo_loss(
    new_logprobs,
    old_logprobs,
    response_mask,
    advantages,
    clip_epsilon: float,
):
    """GRPO-style clipped policy loss, normalized per response.

    This intentionally differs from the DAPO notebook/script in one place:
    DAPO averages over all response tokens globally, while this GRPO baseline
    averages token loss inside each response first and then averages responses.
    """
    old_logprobs = old_logprobs.to(new_logprobs.device)
    shifted_response_mask = response_mask[:, 1:].to(new_logprobs.device)
    advantages = advantages.to(new_logprobs.device).view(-1, 1)

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    token_loss = -torch.minimum(unclipped, clipped) * shifted_response_mask

    response_lengths = shifted_response_mask.sum(dim=1).clamp_min(1.0)
    per_response_loss = token_loss.sum(dim=1) / response_lengths
    valid_responses = shifted_response_mask.sum(dim=1) > 0

    if valid_responses.any():
        return per_response_loss[valid_responses].mean()

    return per_response_loss.mean()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_torchao_compatibility()

    all_groups = read_jsonl(args.rollout_path, limit=args.max_groups)
    grpo_groups = filter_dapo_groups(all_groups)
    print(f"Loaded rollout groups: {len(all_groups)}")
    print(f"GRPO groups after dynamic filtering: {len(grpo_groups)}")

    if not grpo_groups:
        raise ValueError(
            "No GRPO groups left after filtering. Generate more rollouts or use "
            "a rollout file with mixed rewards per prompt."
        )

    tokenizer, policy_model = load_lora_model(
        args.model_name,
        args.adapter_path,
        trainable=True,
    )
    print(f"Trainable parameters: {trainable_parameter_count(policy_model):,}")

    reference_model = None
    if not args.no_reference_model:
        reference_adapter_path = args.reference_adapter_path or args.adapter_path
        _, reference_model = load_lora_model(
            args.model_name,
            reference_adapter_path,
            trainable=False,
        )
        print(f"Using frozen reference adapter: {reference_adapter_path}")
    else:
        print("Using smoke-test mode: old logprobs come from current policy.")

    samples = build_training_samples(
        groups=grpo_groups,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"GRPO training samples: {len(samples)}")

    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
    )

    metrics = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "reference_adapter_path": None
        if args.no_reference_model
        else args.reference_adapter_path or args.adapter_path,
        "rollout_path": args.rollout_path,
        "loaded_groups": len(all_groups),
        "grpo_groups": len(grpo_groups),
        "training_samples": len(samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "epoch_losses": [],
    }

    for epoch in range(args.epochs):
        random.shuffle(samples)
        total_loss = 0.0
        step_count = 0

        for start in tqdm(
            range(0, len(samples), args.batch_size),
            desc=f"epoch {epoch + 1}",
        ):
            batch_samples = samples[start : start + args.batch_size]
            batch = collate_features(batch_samples, tokenizer)

            old_model = policy_model if args.no_reference_model else reference_model
            with torch.no_grad():
                old_logprobs = compute_token_logprobs(
                    old_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                ).detach()

            new_logprobs = compute_token_logprobs(
                policy_model,
                batch["input_ids"],
                batch["attention_mask"],
            )
            loss = grpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs,
                response_mask=batch["response_mask"],
                advantages=batch["advantages"],
                clip_epsilon=args.clip_epsilon,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        metrics["epoch_losses"].append(avg_loss)
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(metrics, output_dir / "grpo_metrics.json")

    print(f"Saved GRPO adapter to {output_dir}")
    print(f"Saved metrics to {output_dir / 'grpo_metrics.json'}")


if __name__ == "__main__":
    main()
