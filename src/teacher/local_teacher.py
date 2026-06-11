from pathlib import Path


def validate_model_name(model_name: str):
    """Give a clear error for placeholder or missing local model paths."""
    model_path = Path(model_name)

    if model_name == "/path/to/deepseek-model":
        raise ValueError(
            "`/path/to/deepseek-model` is only a placeholder. Use a real "
            "Hugging Face repo id like `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` "
            "or a real local folder path."
        )

    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_name}\n"
            "Use a real downloaded model folder, or pass a Hugging Face repo id "
            "such as `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`."
        )


def load_local_teacher(model_name: str):
    """Load a local Hugging Face causal LM as the teacher model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to("cpu")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return tokenizer, model


def format_prompt_for_model(tokenizer, prompt: str) -> str:
    """Use chat formatting when the tokenizer supports it."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def generate_teacher_output(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate one teacher response for one prompt."""
    import torch

    formatted_prompt = format_prompt_for_model(tokenizer, prompt)
    device = next(model.parameters()).device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
