import transformers


def load_tokenizer(model_name: str = "bert-base-chinese"):
    """Load the tokenizer for your model and add some special tokens.

    Args:
        model_name (str): The model name which used as a base model for CWS. \
            The model name need to find in Hugging Face.

    Returns:
        transformers.BertTokenizerFast : A pretrained tokenizer.
    """

    return transformers.BertTokenizerFast.from_pretrained(
        model_name,
        pad_token="[PAD]",
        additional_special_tokens=[
            "[AS]",
            "[CIT]",
            "[MSR]",
            "[PKU]",
            "[UNC]",
            "[CTB6]",
            "[CNC]",
            "[SXU]",
            "[UD]",
            "[WTB]",
            "[ZX]",
        ],
    )
