def load_gsm8k():
    """Load the main GSM8K dataset split collection."""
    from datasets import load_dataset

    # Hugging Face now expects the namespaced dataset repo id.
    return load_dataset("openai/gsm8k", "main")
