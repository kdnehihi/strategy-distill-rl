STRATEGIES = [
    "arithmetic",
    "equation_setup",
    "ratio_or_rate",
    "working_backward",
    "unit_conversion",
    "multi_step_decomposition",
]


def build_base_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show your reasoning clearly, then put only the "
        "final answer inside <answer>...</answer>.\n\n"
        f"Problem:\n{question}"
    )


def build_strategy_teacher_prompt(question: str) -> str:
    strategy_bullets = "\n".join(f"- {s}" for s in STRATEGIES)

    return (
        "You are generating clean training data for a small language model.\n"
        "Solve the math word problem correctly.\n\n"

        "Allowed strategy labels:\n"
        f"{strategy_bullets}\n\n"

        "Instructions:\n"
        "- Prefer starting directly with <final>.\n"
        "- Do not write a long scratchpad before <final>.\n"
        "- Output exactly one <final> block.\n"
        "- The <final> block is mandatory and is the only part that will be parsed.\n"
        "- Inside <strategy>, write exactly one allowed strategy label.\n"
        "- Inside <reasoning>, write concise cleaned-up reasoning, not scratch work.\n"
        "- The <reasoning> text must be at most 3 sentences.\n"
        "- Inside <answer>, write only the final numeric answer.\n"
        "- Do not put words, units, currency symbols, commas, LaTeX, boxed notation, or explanations inside <answer>.\n"
        "- Do not include XML-style tags inside <reasoning>.\n\n"

        "The final block must follow exactly this format:\n"
        "<final>\n"
        "<strategy>one_allowed_label</strategy>\n"
        "<reasoning>concise cleaned-up reasoning</reasoning>\n"
        "<answer>numeric_answer_only</answer>\n"
        "</final>\n\n"

        "Example final block:\n"
        "<final>\n"
        "<strategy>arithmetic</strategy>\n"
        "<reasoning>April sales are 48 clips. May sales are half of April, so 48 / 2 = 24. The total is 48 + 24 = 72.</reasoning>\n"
        "<answer>72</answer>\n"
        "</final>\n\n"

        "Now solve this problem:\n"
        f"{question}"
    )
