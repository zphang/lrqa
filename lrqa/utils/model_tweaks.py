import transformers


def adjust_tokenizer(tokenizer):
    if isinstance(tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
        tokenizer.pad_token = tokenizer.eos_token