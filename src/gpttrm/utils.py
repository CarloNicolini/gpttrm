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
