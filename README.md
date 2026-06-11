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

## Current Scope

This repository currently contains the Week 1 data scaffold plus a small local teacher preview script. It does not include full teacher dataset generation, SFT training, GRPO training, or complex abstractions.

## Future Stages

1. Teacher trace generation
2. Normal distillation
3. Strategy distillation
4. GRPO/RLVR training
5. Evaluation on GSM8K and SVAMP
