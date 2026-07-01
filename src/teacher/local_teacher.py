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


def load_vllm_teacher(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
):
    """Load a vLLM teacher plus tokenizer for prompt formatting."""
    from transformers import AutoTokenizer
    from vllm import LLM

    validate_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    return tokenizer, LLM(**kwargs)


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


def format_prompts_for_model(tokenizer, prompts):
    """Format a list of prompts for chat or base models."""
    return [format_prompt_for_model(tokenizer, prompt) for prompt in prompts]


def generate_teacher_outputs_hf(
    tokenizer,
    model,
    prompts,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
):
    """Generate teacher responses for a batch using Hugging Face Transformers."""
    import torch

    formatted_prompts = format_prompts_for_model(tokenizer, prompts)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(next(model.parameters()).device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
        for output_ids in outputs
    ]


def generate_teacher_outputs_vllm(
    tokenizer,
    llm,
    prompts,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
):
    """Generate teacher responses for a batch using vLLM."""
    from vllm import SamplingParams

    formatted_prompts = format_prompts_for_model(tokenizer, prompts)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n,
    )
    outputs = llm.generate(formatted_prompts, sampling_params)
    if n == 1:
        return [output.outputs[0].text.strip() for output in outputs]

    return [
        [completion.text.strip() for completion in output.outputs]
        for output in outputs
    ]


def generate_teacher_output(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate one teacher response for one prompt."""
    return generate_teacher_outputs_hf(
        tokenizer=tokenizer,
        model=model,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]
