def normalize_answer(ans: str | None) -> str | None:
    """Normalize an answer string for exact-match comparison."""
    if ans is None:
        return None

    normalized = ans.strip().replace(",", "")
    if not normalized:
        return None

    return normalized


def exact_match(pred: str | None, gold: str | None) -> int:
    """Return 1 when normalized prediction and gold answer match exactly."""
    return int(normalize_answer(pred) == normalize_answer(gold))
