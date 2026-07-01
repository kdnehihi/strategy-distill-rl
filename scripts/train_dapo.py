import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_sft_lora import ensure_torchao_compatibility
from src.teacher.local_teacher import load_local_teacher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small offline DAPO-style RLVR adapter from rollout data."
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
        default="data/rl_rollouts_student.jsonl",
        help="Grouped rollout JSONL from scripts/generate_rl_rollouts.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/student_dapo/debug",
        help="Where to save the trained DAPO adapter.",
    )
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int | None = None):
    records = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def write_json(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_dapo_groups(groups: list[dict]) -> list[dict]:
    """Keep only groups with at least two outputs and non-identical rewards."""
    filtered = []
    for group in groups:
        outputs = group.get("outputs", [])
        if len(outputs) < 2:
            continue

        rewards = [output.get("reward") for output in outputs]
        if len(set(rewards)) <= 1:
            continue

        filtered.append(group)

    return filtered


def compute_group_advantages(outputs: list[dict]) -> list[float]:
    rewards = torch.tensor(
        [output["reward"] for output in outputs],
        dtype=torch.float32,
    )
    if rewards.numel() <= 1:
        return [0.0 for _ in outputs]

    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = (rewards - mean) / (std + 1e-6)
    return advantages.tolist()


def tokenize_prompt_response(tokenizer, prompt: str, response: str, max_length: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    input_ids = input_ids[:max_length]
    response_mask = response_mask[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def collate_features(features: list[dict], tokenizer):
    max_len = max(len(feature["input_ids"]) for feature in features)
    pad_id = tokenizer.pad_token_id

    input_ids = []
    attention_mask = []
    response_mask = []
    advantages = []

    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        response_mask.append(feature["response_mask"] + [0] * pad_len)
        advantages.append(feature["advantage"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
    }


def build_training_samples(groups: list[dict], tokenizer, max_length: int):
    samples = []
    for group in groups:
        prompt = group["prompt"]
        outputs = group["outputs"]
        advantages = compute_group_advantages(outputs)

        for output, advantage in zip(outputs, advantages):
            response = output.get("raw_output") or ""
            if not response.strip():
                continue

            feature = tokenize_prompt_response(
                tokenizer=tokenizer,
                prompt=prompt,
                response=response,
                max_length=max_length,
            )
            feature["advantage"] = advantage
            feature["group_id"] = group["id"]
            feature["reward"] = output["reward"]
            feature["is_correct"] = output.get("is_correct", 0)
            feature["is_format_valid"] = output.get("is_format_valid", 0)
            samples.append(feature)

    return samples


def compute_token_logprobs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Token at position t predicts the label at position t+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    return token_logprobs


def dapo_loss(
    new_logprobs,
    old_logprobs,
    response_mask,
    advantages,
    clip_low: float,
    clip_high: float,
):
    shifted_response_mask = response_mask[:, 1:]
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_low,
        max=1.0 + clip_high,
    )

    advantages = advantages.view(-1, 1)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages

    token_loss = -torch.minimum(unclipped, clipped)
    masked_loss = token_loss * shifted_response_mask
    return masked_loss.sum() / shifted_response_mask.sum().clamp_min(1.0)


def load_lora_model(model_name: str, adapter_path: str, trainable: bool):
    from peft import PeftModel

    tokenizer, model = load_local_teacher(model_name)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=trainable,
    )

    if trainable:
        model.train()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    return tokenizer, model


def trainable_parameter_count(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_torchao_compatibility()

    all_groups = read_jsonl(args.rollout_path, limit=args.max_groups)
    dapo_groups = filter_dapo_groups(all_groups)
    print(f"Loaded rollout groups: {len(all_groups)}")
    print(f"DAPO groups after dynamic filtering: {len(dapo_groups)}")

    if not dapo_groups:
        raise ValueError(
            "No DAPO groups left after filtering. Generate more rollouts or use "
            "a rollout file with mixed rewards per prompt."
        )

    tokenizer, policy_model = load_lora_model(
        args.model_name,
        args.adapter_path,
        trainable=True,
    )
    device = next(policy_model.parameters()).device
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
        groups=dapo_groups,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"DAPO training samples: {len(samples)}")

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
        "dapo_groups": len(dapo_groups),
        "training_samples": len(samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
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
            batch = {key: value.to(device) for key, value in batch.items()}

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
            loss = dapo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs,
                response_mask=batch["response_mask"],
                advantages=batch["advantages"],
                clip_low=args.clip_low,
                clip_high=args.clip_high,
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
    write_json(metrics, output_dir / "dapo_metrics.json")

    print(f"Saved DAPO adapter to {output_dir}")
    print(f"Saved metrics to {output_dir / 'dapo_metrics.json'}")


if __name__ == "__main__":
    main()
