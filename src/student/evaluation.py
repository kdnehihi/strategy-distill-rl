from src.evaluation.metrics import exact_match
from src.teacher.formatting import (
    build_canonical_teacher_output,
    check_teacher_format,
    extract_teacher_fields,
    is_valid_teacher_format,
)


def build_student_eval_record(example, raw_output: str):
    """Parse and score one student model output."""
    fields = extract_teacher_fields(raw_output)
    canonical_output = build_canonical_teacher_output(fields)
    is_format_valid = is_valid_teacher_format(raw_output)
    is_correct = exact_match(fields["answer"], example["ground_truth"])

    return {
        "id": example["id"],
        "question": example["question"],
        "ground_truth": example["ground_truth"],
        "raw_model_output": raw_output,
        "model_output": canonical_output,
        "strategy": fields["strategy"],
        "reasoning": fields["reasoning"],
        "model_answer": fields["answer"],
        "is_correct": is_correct,
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
            "format_valid_rate": 0.0,
            "usable_rate": 0.0,
        }

    correct = sum(record["is_correct"] for record in records)
    format_valid = sum(record["is_format_valid"] for record in records)
    usable = sum(record["is_usable"] for record in records)

    return {
        "total": total,
        "correct": correct,
        "format_valid": format_valid,
        "usable": usable,
        "accuracy": correct / total,
        "format_valid_rate": format_valid / total,
        "usable_rate": usable / total,
    }
