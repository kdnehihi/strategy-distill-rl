import re


TARGET_RULES = [
    {
        "name": "money_spent",
        "question_terms": ["spend", "spent", "paid", "pay for", "cost", "price"],
        "output_terms": ["spent", "paid", "cost", "costs", "price", "pay"],
        "conflicting_terms": ["left", "remaining", "profit", "earned", "revenue"],
    },
    {
        "name": "money_profit",
        "question_terms": ["profit", "earn", "earned", "make", "made", "revenue"],
        "output_terms": ["profit", "earn", "earned", "makes", "made", "revenue"],
        "conflicting_terms": ["spent", "paid", "cost", "left", "remaining"],
    },
    {
        "name": "remaining",
        "question_terms": ["left", "remaining", "remain", "how much more", "need"],
        "output_terms": ["left", "remaining", "remain", "more", "needs"],
        "conflicting_terms": ["total", "altogether", "combined", "profit"],
    },
    {
        "name": "total",
        "question_terms": ["total", "altogether", "combined", "in all", "sum"],
        "output_terms": ["total", "altogether", "combined", "in all", "sum"],
        "conflicting_terms": ["left", "remaining", "difference", "profit"],
    },
    {
        "name": "difference",
        "question_terms": ["difference", "how much more", "how much less", "farther"],
        "output_terms": ["difference", "more", "less", "farther"],
        "conflicting_terms": ["total", "altogether", "combined"],
    },
    {
        "name": "rate",
        "question_terms": ["per hour", "per minute", "mph", "speed", "rate"],
        "output_terms": ["per hour", "per minute", "mph", "speed", "rate"],
        "conflicting_terms": ["total distance", "total miles", "total meters"],
    },
]

UNIT_GROUPS = {
    "seconds": {
        "question_terms": ["second", "seconds"],
        "output_terms": ["second", "seconds", "sec"],
        "conflicting_terms": ["minute", "minutes", "hour", "hours"],
    },
    "minutes": {
        "question_terms": ["minute", "minutes"],
        "output_terms": ["minute", "minutes", "min"],
        "conflicting_terms": ["second", "seconds", "hour", "hours"],
    },
    "hours": {
        "question_terms": ["hour", "hours"],
        "output_terms": ["hour", "hours"],
        "conflicting_terms": ["second", "seconds", "minute", "minutes"],
    },
    "years": {
        "question_terms": ["year", "years", "per year"],
        "output_terms": ["year", "years", "yearly", "per year"],
        "conflicting_terms": ["month", "months", "monthly", "week", "weeks"],
    },
    "months": {
        "question_terms": ["month", "months", "per month"],
        "output_terms": ["month", "months", "monthly", "per month"],
        "conflicting_terms": ["year", "years", "yearly", "week", "weeks"],
    },
    "dollars": {
        "question_terms": ["dollar", "dollars", "$", "money", "cost", "paid", "profit"],
        "output_terms": ["dollar", "dollars", "$", "money"],
        "conflicting_terms": ["minutes", "hours", "miles", "meters", "pages"],
    },
    "miles": {
        "question_terms": ["mile", "miles", "mph"],
        "output_terms": ["mile", "miles", "mph"],
        "conflicting_terms": ["meter", "meters", "kilometer", "kilometers"],
    },
    "meters": {
        "question_terms": ["meter", "meters"],
        "output_terms": ["meter", "meters"],
        "conflicting_terms": ["mile", "miles", "kilometer", "kilometers"],
    },
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def infer_target_rules(question: str) -> list[dict]:
    question_text = normalize_text(question)
    return [
        rule
        for rule in TARGET_RULES
        if contains_any(question_text, rule["question_terms"])
    ]


def infer_target_units(question: str) -> list[tuple[str, dict]]:
    question_text = normalize_text(question)
    return [
        (unit_name, unit_rule)
        for unit_name, unit_rule in UNIT_GROUPS.items()
        if contains_any(question_text, unit_rule["question_terms"])
    ]


def target_quantity_score(question: str, output_text: str) -> tuple[float, dict]:
    """Heuristic score for whether the response addresses the asked quantity.

    This is intentionally conservative. It gives a small bonus when the
    response mentions the same target quantity/unit as the question and applies
    a small penalty for obvious unit or target mismatches.
    """
    output_text = normalize_text(output_text)
    target_rules = infer_target_rules(question)
    target_units = infer_target_units(question)

    matched_targets = [
        rule["name"]
        for rule in target_rules
        if contains_any(output_text, rule["output_terms"])
    ]
    conflicting_targets = [
        rule["name"]
        for rule in target_rules
        if contains_any(output_text, rule["conflicting_terms"])
        and not contains_any(output_text, rule["output_terms"])
    ]

    matched_units = [
        unit_name
        for unit_name, unit_rule in target_units
        if contains_any(output_text, unit_rule["output_terms"])
    ]
    unit_mismatches = [
        unit_name
        for unit_name, unit_rule in target_units
        if contains_any(output_text, unit_rule["conflicting_terms"])
        and not contains_any(output_text, unit_rule["output_terms"])
    ]

    target_reward = 0.2 if (matched_targets or matched_units) else 0.0
    unit_mismatch_penalty = -0.2 if (conflicting_targets or unit_mismatches) else 0.0
    score = target_reward + unit_mismatch_penalty

    return score, {
        "target_rules": [rule["name"] for rule in target_rules],
        "target_units": [unit_name for unit_name, _ in target_units],
        "matched_targets": matched_targets,
        "matched_units": matched_units,
        "conflicting_targets": conflicting_targets,
        "unit_mismatches": unit_mismatches,
        "target_quantity_reward": target_reward,
        "unit_mismatch_penalty": unit_mismatch_penalty,
    }


def score_student_output(example: dict, eval_record: dict) -> tuple[float, dict]:
    correctness_reward = 1.0 if eval_record["is_correct"] else 0.0
    format_reward = 0.1 if eval_record["is_format_valid"] else -0.2
    output_text = " ".join(
        str(eval_record.get(key) or "")
        for key in ["model_output", "raw_model_output", "reasoning", "model_answer"]
    )
    target_score, target_details = target_quantity_score(
        question=example["question"],
        output_text=output_text,
    )

    reward = correctness_reward + format_reward + target_score
    details = {
        "correctness_reward": correctness_reward,
        "format_reward": format_reward,
        **target_details,
    }
    return round(reward, 4), details
