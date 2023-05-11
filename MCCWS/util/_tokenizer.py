import transformers


def load_tokenizer(model_name, additional_token: list = []):
  '''
    Load the tokenizer for your model
    ------------------------------------------
    BertTokenizerFast is for bert -> PreTrainedTokenizerFast can use in general case
  '''
  return transformers.BertTokenizerFast.from_pretrained(
    model_name,
    pad_token='[PAD]',
    additional_special_tokens=[
      "[AS]",
      "[CIT]",
      "[MSR]",
      "[PKU]",
      "[UNC]",
      '[CTB6]',
      '[CNC]',
      '[SXU]',
      '[UD]',
      '[WTB]',
      '[ZX]',
    ],
  )
