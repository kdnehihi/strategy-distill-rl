import re

from src.evaluation.metrics import exact_match
from src.teacher.formatting import (
    build_canonical_teacher_output,
    check_teacher_format,
    extract_teacher_fields,
    is_valid_teacher_format,
)
from src.utils.parsers import extract_tagged_answer, normalize_numeric_answer


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_PATTERN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
ANSWER_PHRASE_PATTERN = re.compile(
    r"(?:answer|therefore|so)[^-\d$]{0,40}(-?\$?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def extract_loose_numeric_answer(raw_output: str | None) -> str | None:
    """Best-effort numeric answer extraction for baseline diagnostics only.

    Strict training/eval format still requires the <final> block. This helper is
    only used to estimate whether a zero-shot model did the math correctly when
    it ignored the XML output format.
    """
    if not raw_output:
        return None

    tagged_answer = normalize_numeric_answer(extract_tagged_answer(raw_output))
    if tagged_answer:
        return tagged_answer

    boxed_matches = BOXED_PATTERN.findall(raw_output)
    if boxed_matches:
        return normalize_numeric_answer(boxed_matches[-1])

    phrase_matches = ANSWER_PHRASE_PATTERN.findall(raw_output)
    if phrase_matches:
        return normalize_numeric_answer(phrase_matches[-1])

    number_matches = NUMBER_PATTERN.findall(raw_output)
    if number_matches:
        return normalize_numeric_answer(number_matches[-1])

    return None


def build_student_eval_record(example, raw_output: str):
    """Parse and score one student model output."""
    fields = extract_teacher_fields(raw_output)
    canonical_output = build_canonical_teacher_output(fields)
    is_format_valid = is_valid_teacher_format(raw_output)
    is_correct = exact_match(fields["answer"], example["ground_truth"])
    loose_answer = extract_loose_numeric_answer(raw_output)
    is_loose_correct = exact_match(loose_answer, example["ground_truth"])

    return {
        "id": example["id"],
        "question": example["question"],
        "ground_truth": example["ground_truth"],
        "raw_model_output": raw_output,
        "model_output": canonical_output,
        "strategy": fields["strategy"],
        "reasoning": fields["reasoning"],
        "model_answer": fields["answer"],
        "loose_model_answer": loose_answer,
        "is_correct": is_correct,
        "is_loose_correct": is_loose_correct,
        "is_format_valid": int(is_format_valid),
        "is_usable": int(is_correct and is_format_valid),
        "format_checks": check_teacher_format(raw_output),
    }


def summarize_eval_records(records):
    """Return aggregate metrics for student evaluation records."""
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "loose_math_accuracy": 0.0,
            "format_valid_rate": 0.0,
            "usable_rate": 0.0,
        }

    correct = sum(record["is_correct"] for record in records)
    loose_correct = sum(record["is_loose_correct"] for record in records)
    format_valid = sum(record["is_format_valid"] for record in records)
    usable = sum(record["is_usable"] for record in records)

    return {
        "total": total,
        "correct": correct,
        "loose_correct": loose_correct,
        "format_valid": format_valid,
        "usable": usable,
        "accuracy": correct / total,
        "loose_math_accuracy": loose_correct / total,
        "format_valid_rate": format_valid / total,
        "usable_rate": usable / total,
    }
