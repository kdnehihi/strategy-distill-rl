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

## Current Scope

This repository currently contains only the Week 1 scaffold. It does not include teacher API generation, SFT training, GRPO training, or complex abstractions.

## Future Stages

1. Teacher trace generation
2. Normal distillation
3. Strategy distillation
4. GRPO/RLVR training
5. Evaluation on GSM8K and SVAMP
