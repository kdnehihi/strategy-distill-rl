import re


def extract_gsm8k_answer(answer_text: str) -> str | None:
    """Extract the final answer after GSM8K's #### marker."""
    marker = "####"
    if marker not in answer_text:
        return None

    answer = answer_text.split(marker, maxsplit=1)[1].strip()
    if not answer:
        return None

    return answer.replace(",", "")


def extract_tagged_answer(model_output: str) -> str | None:
    """Extract text inside <answer>...</answer> from a final block."""
    match = re.search(r"<answer>(.*?)</answer>", model_output, flags=re.DOTALL)
    if match is None:
        return None

    answer = match.group(1).strip()
    if not answer:
        return None

    return answer.replace(",", "")


def extract_tagged_text(model_output: str, tag: str) -> str | None:
    """Extract text inside a simple XML-style tag."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, model_output, flags=re.DOTALL)
    if match is None:
        return None

    text = match.group(1).strip()
    return text or None


def extract_final_block(model_output: str) -> str | None:
    """Extract only the content inside a complete <final>...</final> block."""
    return extract_tagged_text(model_output, "final")


def extract_reasoning(model_output: str) -> str | None:
    """Extract text inside <reasoning>...</reasoning>."""
    return extract_tagged_text(model_output, "reasoning")


def extract_strategy(model_output: str) -> str | None:
    """Extract text inside <strategy>...</strategy> from a final block."""
    match = re.search(r"<strategy>(.*?)</strategy>", model_output, flags=re.DOTALL)
    if match is None:
        return None

    strategy = match.group(1).strip()
    if not strategy:
        return None

    return strategy


def normalize_numeric_answer(answer: str | None) -> str | None:
    """
    Basic normalization for numeric answers.
    Example:
    '1,200' -> '1200'
    '$72' -> '72'
    '72.' -> '72'
    """
    if answer is None:
        return None

    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("\\$", "$")
    answer = answer.replace("$", "")
    answer = answer.strip()

    # Remove trailing period if answer is like "72."
    if answer.endswith("."):
        answer = answer[:-1]

    return answer
