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
    strategy_list = "\n".join(f"- {strategy}" for strategy in STRATEGIES)
    return (
        "You are a math reasoning teacher. Choose exactly one strategy from the "
        "list below, solve the problem, and use this exact output format:\n\n"
        "<strategy>chosen_strategy</strategy>\n"
        "<think>your reasoning</think>\n"
        "<answer>final answer</answer>\n\n"
        "Strategies:\n"
        f"{strategy_list}\n\n"
        f"Problem:\n{question}"
    )
