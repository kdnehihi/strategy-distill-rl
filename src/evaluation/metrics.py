from src.utils.parsers import normalize_numeric_answer


def normalize_answer(ans: str | None) -> str | None:
    """Normalize an answer string for exact-match comparison."""
    return normalize_numeric_answer(ans)


def exact_match(pred: str | None, gold: str | None) -> int:
    """Return 1 when normalized prediction and gold answer match exactly."""
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    if pred is None or gold is None:
        return 0

    return int(pred == gold)
