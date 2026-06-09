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
    """Extract text inside <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", model_output, flags=re.DOTALL)
    if match is None:
        return None

    answer = match.group(1).strip()
    if not answer:
        return None

    return answer.replace(",", "")


def extract_strategy(model_output: str) -> str | None:
    """Extract text inside <strategy>...</strategy>."""
    match = re.search(r"<strategy>(.*?)</strategy>", model_output, flags=re.DOTALL)
    if match is None:
        return None

    strategy = match.group(1).strip()
    if not strategy:
        return None

    return strategy
