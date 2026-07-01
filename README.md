# Strategy-Distill-RL

Strategy-Distill-RL is a small machine learning research project for studying strategy-level distillation and GRPO/RLVR for small LLM mathematical reasoning.

## Week 1 Goal

The Week 1 goal is to build a minimal data and utility scaffold around GSM8K:

- Load GSM8K
- Inspect the dataset
- Extract ground-truth answers
- Create cleaned JSONL files
- Define prompt templates for future teacher and student use
- Add basic evaluation and parser utilities

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare GSM8K

From the project root, run:

```bash
python scripts/prepare_gsm8k.py
```

This creates:

- `data/gsm8k_clean_train.jsonl`
- `data/gsm8k_clean_test.jsonl`

## Teacher Preview

After preparing GSM8K, run a small local teacher preview:

```bash
python scripts/generate_teacher_preview.py --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --num-samples 10
```

If the model is already downloaded locally, pass the real local folder path instead of the Hugging Face repo id.

This creates:

- `data/gsm8k_teacher_preview.jsonl`

Each record includes the teacher output, parsed strategy, parsed answer, and whether the answer exactly matches the GSM8K ground truth.

To check teacher output formatting:

```bash
python scripts/check_teacher_preview.py
```

For the larger teacher dataset, run:

```bash
python scripts/validate_teacher_dataset.py --path data/gsm8k_teacher_preview_5000.jsonl
```

## Student Baseline and SFT

Run a zero-shot student baseline first:

```bash
python scripts/evaluate_student.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --input-path data/gsm8k_clean_test.jsonl \
  --num-samples 100 \
  --output-path runs/eval/qwen25_math_1p5b_zero_shot.jsonl \
  --metrics-path runs/eval/qwen25_math_1p5b_zero_shot_metrics.json
```

The evaluator reports strict `accuracy`, `format_valid_rate`, and
`usable_rate`. It also reports diagnostic `loose_math_accuracy`, which extracts
a best-effort numeric answer from raw model text for zero-shot analysis only.
Use `usable_rate` as the main pipeline metric.

Build SFT files from usable teacher traces:

```bash
python scripts/build_sft_dataset.py \
  --teacher-path data/gsm8k_teacher_preview_5000.jsonl
```

Train a lightweight LoRA SFT adapter:

```bash
python scripts/train_sft_lora.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --train-path data/sft_strategy_train.jsonl \
  --val-path data/sft_strategy_val.jsonl \
  --max-train-samples 300 \
  --max-val-samples 100
```

Evaluate the trained adapter with the same evaluator:

```bash
python scripts/evaluate_student.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-path checkpoints/qwen25_math_1p5b_strategy_lora \
  --input-path data/gsm8k_clean_test.jsonl \
  --num-samples 100 \
  --output-path runs/eval/qwen25_math_1p5b_lora.jsonl \
  --metrics-path runs/eval/qwen25_math_1p5b_lora_metrics.json
```

The same workflow is available as a notebook:

- `notebooks/03_student_sft_experiments.ipynb`

## RLVR Rollout Generation

After choosing an SFT adapter, generate multiple sampled answers per GSM8K
prompt before implementing DAPO/RLVR training:

```bash
python scripts/generate_rl_rollouts.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --input-path data/gsm8k_clean_train.jsonl \
  --output-path data/rl_rollouts_debug.jsonl \
  --num-samples 50 \
  --num-generations 4 \
  --batch-size 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 512
```

Each output group keeps the question, ground truth, prompt, sampled outputs,
parsed answers, strict format checks, correctness flags, and a simple reward.
Start with 4 generations per prompt for debugging, then increase to 8 once the
rollout quality and runtime look reasonable.

## Current Scope

This repository currently contains the Week 1 data scaffold, teacher trace generation, teacher validation, zero-shot student evaluation, and lightweight LoRA SFT. It does not include GRPO/RLVR training or complex abstractions.

## Future Stages

1. Teacher trace generation
2. Normal distillation
3. Strategy distillation
4. GRPO/RLVR training
5. Evaluation on GSM8K and SVAMP
