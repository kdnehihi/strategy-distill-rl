def load_gsm8k():
    """Load the main GSM8K dataset split collection."""
    from datasets import load_dataset

    try:
        # Newer Hugging Face tooling expects the namespaced dataset repo id.
        return load_dataset("openai/gsm8k", "main")
    except Exception:
        # Older environments may still resolve the legacy short name.
        return load_dataset("gsm8k", "main")
