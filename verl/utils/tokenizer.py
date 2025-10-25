"""Utils for tokenization."""

import warnings
import transformers
from transformers import AutoTokenizer

__all__ = ["hf_tokenizer", "hf_processor"]

def hf_tokenizer(name_or_path: str, correct_pad_token: bool = True, correct_gemma2: bool = True, **kwargs) -> transformers.PreTrainedTokenizer:

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer

def set_pad_token_id(tokenizer: transformers.PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)

def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
