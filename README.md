# Strategy-Distill-RL

Strategy-Distill-RL is a small machine learning research project for studying
strategy-level distillation and RLVR for small LLM mathematical reasoning.

The core question is:

> Can a small math LLM learn a stable structured reasoning format from teacher
> traces, then improve answer correctness with RLVR without losing that format?

The current project focuses on GSM8K with Qwen2.5-Math students. Teacher traces
are used to teach a structured output schema:

```xml
<final>
<strategy>...</strategy>
<reasoning>...</reasoning>
<answer>...</answer>
</final>
```

The main result so far is that DAPO-style RLVR improves the 1.5B SFT student
while preserving the structured output format.

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

## DAPO and GRPO RLVR

Generate rollouts from the SFT student:

```bash
python scripts/generate_rl_rollouts.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --input-path data/gsm8k_clean_train.jsonl \
  --output-path data/rl_rollouts_1p5b_sft_g8.jsonl \
  --num-samples -1 \
  --num-generations 8 \
  --batch-size 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 512
```

Train DAPO-style RLVR:

```bash
python scripts/train_dapo.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --reference-adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --rollout-path data/rl_rollouts_1p5b_sft_g8.jsonl \
  --output-dir checkpoints/student_dapo/dapo_1p5b_g8_lr5e7_e1 \
  --max-groups 100000 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --clip-low 0.2 \
  --clip-high 0.28 \
  --max-length 1024
```

Train a GRPO-style baseline on the same rollouts:

```bash
python scripts/train_grpo.py \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --reference-adapter-path checkpoints/student_sft/balanced_r16_a32_4000 \
  --rollout-path data/rl_rollouts_1p5b_sft_g8.jsonl \
  --output-dir checkpoints/student_grpo/grpo_1p5b_g8_lr5e7_e1 \
  --max-groups 100000 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --clip-epsilon 0.2 \
  --max-length 1024
```

Notebook entry points:

- `notebooks/04_dapo_training.ipynb`: 1.5B DAPO training from SFT rollouts.
- `notebooks/06_evaluate_dapo_checkpoint.ipynb`: restore a DAPO checkpoint,
  evaluate it, and run paired error analysis.
- `notebooks/07_grpo_training.ipynb`: train GRPO on the same rollout data and
  compare SFT vs DAPO vs GRPO.

## Current Results

Full GSM8K test set has 1,319 examples. Metrics below use the same local
evaluator and strict structured-output parser.

| Model | Stage | Accuracy | Loose Math Accuracy | Format Valid | Usable |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Math-1.5B | SFT | 70.28% | 70.36% | 99.47% | 70.28% |
| Qwen2.5-Math-1.5B | DAPO, 8 rollouts | 74.22% | 74.45% | 99.39% | 74.22% |
| Qwen2.5-Math-7B | Zero-shot | 0.00% strict | 40.41% loose | 0.00% | 0.00% |
| Qwen2.5-Math-7B | SFT | 82.64% | 82.64% | 99.70% | 82.64% |

The 1.5B DAPO checkpoint improved over the 1.5B SFT checkpoint by:

- `+52` net correct answers on GSM8K test.
- `87` SFT-wrong examples fixed by DAPO.
- `35` SFT-correct examples regressed by DAPO.
- Structured output validity remained above `99%`.

This suggests that SFT established the output format as a stable policy prior,
while DAPO improved answer correctness inside that learned format.

## Observations

### 1. SFT Learns the Output Manifold

Zero-shot models often solve some math but do not follow the required XML-like
schema. After SFT, the student reliably produces:

```xml
<final>
<strategy>...</strategy>
<reasoning>...</reasoning>
<answer>...</answer>
</final>
```

This matters because DAPO does not explicitly train a separate format loss. It
works on sampled model outputs. If SFT did not already make the format stable,
RLVR would waste reward signal on formatting instead of math reasoning.

### 2. DAPO Improves Correctness Without Breaking Format

The DAPO reward is answer-centric, but format validity stayed above `99%`.
That is a useful finding:

> SFT creates a stable structured-output prior; DAPO adjusts answer correctness
> while mostly staying within that prior.

This is the main reason the project uses an SFT -> RLVR pipeline instead of
jumping directly into RL.

### 3. More Rollouts Are Not Automatically Better

Increasing rollouts from 4 to 8 gave only a small final accuracy improvement.
The main bottleneck is not just the number of sampled outputs. Useful RL signal
comes from groups where the same question has mixed rewards: at least one good
and one bad rollout. If all rollouts are correct or all are wrong, group-relative
advantages carry little learning signal.

### 4. Remaining Errors Are Semantic, Not Formatting Errors

Most DAPO errors still have valid XML tags and numeric answers. The remaining
failures are usually reasoning failures:

- Wrong percentage base.
- Missing multiplicative factor.
- Confusing a choice with a sum.
- Direction or remaining-distance mistakes.
- Off-by-one or strict-inequality cases.
- Incorrect reverse equation setup.
- Mixture/proportional reasoning errors.

Example pattern:

```text
Question asks for the better of two investment choices.
Model computes both profits correctly but adds them together.
```

This shows that RLVR improved selection among existing behaviors, but did not
fully solve deeper semantic interpretation.

### 5. 7B Is Useful as a Scale-Up Ablation, Not the Main Story

Qwen2.5-Math-7B SFT performs much better than 1.5B SFT, but DAPO training for
7B was not compute-efficient in the current Colab setup. The model required
CPU/GPU offloading and training steps became prohibitively slow.

For this project, the 1.5B setting is more valuable:

- It is small enough for affordable iteration.
- It has enough errors for RLVR gains to be visible.
- It better demonstrates the role of distillation and DAPO.

## Failure Modes and Possible Fixes

| Failure Mode | Why It Happens | Possible Fix |
|---|---|---|
| Wrong percentage base | Reward only checks final answer, not whether the base quantity was selected correctly. | Add targeted teacher data or verifier checks for percent-base problems. |
| Missing repeated factor | Model reads one event but ignores frequency such as "3 times a week". | Add error-tagged examples and evaluate by question type. |
| Choice vs sum confusion | Model computes all options and aggregates instead of selecting max/min. | Add strategy-specific reward or prompt constraints for comparison problems. |
| Off-by-one break-even errors | Exact-match reward cannot explain strict "starts earning" vs "breaks even". | Add a reasoning verifier or rubric reward for inequality semantics. |
| Direction/remaining-distance error | Model computes total traveled instead of distance remaining. | Add more working-backward and remaining-distance examples. |
| Long reasoning still wrong | Format is correct, but internal logic has a hidden semantic error. | Add process-level reward, LLM-as-judge scoring, or symbolic checks for selected templates. |

## Future Improvements

1. **Question-type error analysis**
   Add lightweight classifiers for error types such as percentage, rate, mixture,
   comparison, reverse reasoning, and remaining-distance problems.

2. **Hybrid reward**
   Combine exact-match reward with small process rewards:
   - valid format
   - allowed strategy
   - answer only numeric
   - no text after `</final>`
   - optional LLM judge score for reasoning consistency

3. **LLM-as-a-judge reward**
   Use a judge model to score reasoning quality, especially for unlabeled or
   hard-to-verify data. This would complement the current verifiable GSM8K
   reward rather than replace it immediately.

4. **Better rollout filtering**
   Keep only groups with reward variance and inspect the distribution of
   correct/incorrect samples per prompt before RL training.

5. **DAPO vs GRPO comparison**
   Train GRPO with the same SFT checkpoint, rollout data, and evaluation script.
   Compare:
   - final accuracy
   - format validity
   - paired fixes/regressions
   - cost per improvement

6. **Report-ready artifacts**
   Save all experiment metrics to CSV/JSON and write a final report with:
   - SFT vs DAPO metrics
   - paired comparison
   - rollout variance statistics
   - representative fixes and regressions
   - cost/compute notes

## Current Scope

This repository currently contains:

- GSM8K data preparation.
- Teacher trace generation and validation.
- Structured prompt/parsing utilities.
- Student zero-shot and adapter evaluation.
- LoRA SFT training.
- RL rollout generation.
- Offline DAPO-style RLVR training.
- Offline GRPO-style comparison training.
- DAPO checkpoint evaluation and paired error analysis.

It intentionally avoids complicated abstractions so that each research step is
easy to inspect and modify.

## Future Stages

1. Final experiment report with tables and representative cases.
2. Question-type and failure-mode analysis.
3. GRPO comparison against the current DAPO checkpoint.
4. Hybrid exact-match plus reasoning-quality reward.
5. Optional LLM-as-a-judge reward experiments.
6. Evaluation on SVAMP or another out-of-distribution math set.
