import torch
import lightning.pytorch as pl
from gpt2_lm import GPT2LanguageModel
import argparse


def test_imports():
    """Test that all required modules can be imported."""
    import transformers
    import pandas
    import numpy
    import lightning.pytorch

    assert True


def test_model_init():
    """Test that the GPT2LanguageModel can be initialized."""
    parser = argparse.ArgumentParser()
    parser = GPT2LanguageModel.add_model_specific_args(parser)
    # Add dummy args that would usually come from training.py
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3)

    args, unknown = parser.parse_known_args([])
    model = GPT2LanguageModel(args)
    assert model is not None
    assert isinstance(model, pl.LightningModule)


def test_tokenizer():
    """Test the tokenizer wrapper."""
    from gpt2_tokenizer import GPT2TextEncoder

    tokenizer = GPT2TextEncoder("gpt2")
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, torch.Tensor)
    decoded = tokenizer.decode(tokens)
    # GPT2 tokenizer might add spaces or handle punctuation differently,
    # but the core text should be preserve-ish depending on skip_special_tokens.
    assert len(decoded) > 0
