import re

from src.utils.parsers import (
    extract_final_block,
    extract_reasoning,
    extract_strategy,
    extract_tagged_answer,
    normalize_numeric_answer,
)
from src.utils.prompts import STRATEGIES


XML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")


def is_numeric_answer(answer: str | None) -> bool:
    """Return True when answer is already a plain numeric value.

    This is intentionally stricter than normalize_numeric_answer: "$5" can be
    normalized for comparison, but it is not valid teacher-output format.
    """
    if answer is None:
        return False

    answer = answer.strip()
    return re.fullmatch(r"-?\d+(?:\.\d+)?", answer) is not None


def count_final_blocks(raw_output: str) -> int:
    """Count complete final blocks without repairing incomplete blocks."""
    return len(re.findall(r"<final>.*?</final>", raw_output, flags=re.DOTALL))


def reasoning_has_no_xml_tags(reasoning: str | None) -> bool:
    """Reject nested XML-style tags inside cleaned reasoning."""
    if reasoning is None:
        return False

    return XML_TAG_PATTERN.search(reasoning) is None


def extract_teacher_fields(raw_output: str):
    """Extract teacher fields only from a complete final block.

    We use <reasoning> instead of model-specific reasoning tags because some
    reasoning models have special habits around those tags.
    """
    final_block = extract_final_block(raw_output)
    if final_block is None:
        # A complete final block is required so truncated scratchpad text cannot
        # become fake teacher answers like "2", "3", or "18".
        return {
            "strategy": None,
            "reasoning": None,
            "answer": None,
        }

    strategy = extract_strategy(final_block)
    reasoning = extract_reasoning(final_block)
    answer = normalize_numeric_answer(extract_tagged_answer(final_block))

    if strategy not in STRATEGIES:
        strategy = None

    return {
        "strategy": strategy,
        "reasoning": reasoning,
        "answer": answer,
    }


def build_canonical_teacher_output(fields) -> str | None:
    """Build the cleaned final block once all required fields exist."""
    if (
        fields["strategy"] is None
        or fields["reasoning"] is None
        or fields["answer"] is None
    ):
        return None

    return (
        "<final>\n"
        f"<strategy>{fields['strategy']}</strategy>\n"
        f"<reasoning>{fields['reasoning']}</reasoning>\n"
        f"<answer>{fields['answer']}</answer>\n"
        "</final>"
    )


def check_teacher_format(raw_output: str | None):
    """Return a checklist for the required teacher final block format."""
    if raw_output is None:
        return {
            "has_final_block": False,
            "has_exactly_one_final_block": False,
            "has_strategy_tag": False,
            "strategy_is_allowed": False,
            "has_reasoning_tag": False,
            "has_answer_tag": False,
            "answer_only_number": False,
            "reasoning_non_empty": False,
            "reasoning_has_no_xml_tags": False,
            "no_text_after_final": False,
        }

    final_block = extract_final_block(raw_output)
    final_block_count = count_final_blocks(raw_output)
    if final_block is None:
        # Do not use raw-text number fallback for teacher outputs. The final
        # block is the only trusted parsed region.
        return {
            "has_final_block": False,
            "has_exactly_one_final_block": False,
            "has_strategy_tag": False,
            "strategy_is_allowed": False,
            "has_reasoning_tag": False,
            "has_answer_tag": False,
            "answer_only_number": False,
            "reasoning_non_empty": False,
            "reasoning_has_no_xml_tags": False,
            "no_text_after_final": False,
        }

    strategy = extract_strategy(final_block)
    reasoning = extract_reasoning(final_block)
    raw_answer = extract_tagged_answer(final_block)

    return {
        "has_final_block": True,
        "has_exactly_one_final_block": final_block_count == 1,
        "has_strategy_tag": strategy is not None,
        "strategy_is_allowed": strategy in STRATEGIES,
        "has_reasoning_tag": reasoning is not None,
        "has_answer_tag": raw_answer is not None,
        "answer_only_number": is_numeric_answer(raw_answer),
        "reasoning_non_empty": reasoning is not None and bool(reasoning.strip()),
        "reasoning_has_no_xml_tags": reasoning_has_no_xml_tags(reasoning),
        "no_text_after_final": raw_output.strip().endswith("</final>"),
    }


def is_valid_teacher_format(raw_output: str | None) -> bool:
    """Return True only when the raw output has a complete valid final block."""
    checks = check_teacher_format(raw_output)
    return all(checks.values())
