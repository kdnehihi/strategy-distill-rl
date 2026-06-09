def load_gsm8k():
    """Load the main GSM8K dataset split collection."""
    from datasets import load_dataset

    return load_dataset("gsm8k", "main")
