# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def top_k_top_p_filtering(
    logits, top_k, top_p, temperature, filter_value=-float("Inf")
):
    # Hugging Face script to apply top k and nucleus sampling
    logits = logits / temperature

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def load_weights_lm_head(model, original_model):
    """
    Transfer weights from the GPT2LMHeadModel to our model.
    """
    original_dict = original_model.state_dict()
    model_dict = model.state_dict()

    new_state_dict = {}

    # Map original transformer weights to self.gpt2
    for k, v in original_dict.items():
        if k.startswith("transformer."):
            new_key = k.replace("transformer.", "gpt2.")
            if new_key in model_dict:
                # Check for size mismatch (e.g. embeddings)
                if v.shape == model_dict[new_key].shape:
                    new_state_dict[new_key] = v
                else:
                    # For embeddings, we copy what we can and keep the rest random/zero
                    print(f"Size mismatch for {k}, copying partial weights.")
                    new_state_dict[new_key] = model_dict[new_key].clone()
                    min_0 = min(v.shape[0], model_dict[new_key].shape[0])
                    if len(v.shape) > 1:
                        new_state_dict[new_key][:min_0, :] = v[:min_0, :]
                    else:
                        new_state_dict[new_key][:min_0] = v[:min_0]

        elif k == "lm_head.weight":
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"Size mismatch for {k}, copying partial weights.")
                    new_state_dict[k] = model_dict[k].clone()
                    min_size = min(v.shape[0], model_dict[k].shape[0])
                    new_state_dict[k][:min_size, :] = v[:min_size, :]

    model.load_state_dict(new_state_dict, strict=False)


import math

import torch
from torch import nn


def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
