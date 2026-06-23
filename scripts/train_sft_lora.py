import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.prompts import build_strategy_student_prompt


def read_jsonl(path: Path, limit: int | None = None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def format_prompt_for_model(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def tokenize_sft_record(tokenizer, record, max_length: int):
    prompt = build_strategy_student_prompt(record["question"])
    formatted_prompt = format_prompt_for_model(tokenizer, prompt)
    target = record["target"].strip()

    full_text = formatted_prompt + target + tokenizer.eos_token
    prompt_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
    )["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_data_collator(tokenizer):
    import torch

    def collate(features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def make_training_args(**kwargs):
    import inspect
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    return TrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple LoRA SFT student on usable teacher traces."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base student model name or local path.",
    )
    parser.add_argument(
        "--train-path",
        default="data/sft_strategy_train.jsonl",
        help="SFT train JSONL from scripts/build_sft_dataset.py.",
    )
    parser.add_argument(
        "--val-path",
        default="data/sft_strategy_val.jsonl",
        help="SFT validation JSONL from scripts/build_sft_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/qwen25_math_1p5b_strategy_lora",
        help="Directory for LoRA adapter checkpoints.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(Path(args.train_path), args.max_train_samples)
    val_records = read_jsonl(Path(args.val_path), args.max_val_samples)

    train_dataset = Dataset.from_list(train_records).map(
        lambda record: tokenize_sft_record(tokenizer, record, args.max_length),
        remove_columns=list(train_records[0].keys()),
    )
    val_dataset = Dataset.from_list(val_records).map(
        lambda record: tokenize_sft_record(tokenizer, record, args.max_length),
        remove_columns=list(val_records[0].keys()),
    )

    training_args = make_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=build_data_collator(tokenizer),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
        "model_name": args.model_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    }
    metrics_path = Path(args.output_dir) / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved LoRA adapter to {args.output_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
