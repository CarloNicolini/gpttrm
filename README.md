# GPTTRM — GPT-2 with Tiny Recursive Model Integration

> **Research codebase** exploring how [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) (TRM) can be plugged into a standard GPT-2 transformer to add iterative latent reasoning at training time.

---

## Research Motivation

Modern large language models are essentially **single-pass** function evaluators: tokens flow through a fixed stack of transformer layers and produce a distribution over the vocabulary in one shot. While scaling depth and width has driven remarkable progress, several lines of evidence suggest that **iterative, recursive computation** — letting the model "think longer" on harder inputs — can unlock capabilities that raw scale alone cannot:

| Insight | Source |
|---|---|
| Intermediate transformer layers carry the highest intrinsic dimension and representational capacity | Aghajanyan et al. (2020), *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* |
| Adaptive Computation Time (ACT) allows networks to allocate variable compute per token | Graves (2016), *Adaptive Computation Time for RNNs* |
| Test-time compute scaling ("thinking longer") materially improves reasoning | Snell et al. (2024), *Scaling LLM Test-Time Compute*; OpenAI o1/o3 |
| Tiny networks with deep recursion and deep supervision generalize better than large single-pass models on reasoning tasks | Martineaux (2025), *Less is More: Recursive Reasoning with Tiny Networks* |

The **Tiny Recursive Model (TRM)** paper by Alexia Joëlicaux Martineaux proposes a particularly elegant instantiation of this idea. Instead of two separate networks (as in the predecessor Hierarchical Reasoning Model), TRM uses a **single tiny network** (2 layers) applied recursively with two interleaved latent variables:

- **z (latent reasoning)**: a chain-of-thought–like hidden state that accumulates reasoning across recursion steps.
- **y (proposed solution)**: the current best guess at the output, updated less frequently.

The recursion loop runs `T` outer cycles × `n` inner cycles, with `T-1` cycles executed **without gradients** (cheap forward-only pondering) and only the final cycle backpropagated — a form of implicit deep supervision that avoids the memory cost of full BPTT while still benefiting from the iterated computation.

### Why GPT-2?

GPT-2 is the simplest modern autoregressive transformer that is still architecturally representative of larger models (GPT-3/4, LLaMA, etc.). By building a **custom from-scratch GPT-2** with controllable layer count and hidden dimension, we can:

1. Train from random initialization (no confound from pre-trained weights).
2. Precisely control where in the network the TRM block is inserted.
3. Keep the total parameter count small enough for rapid iteration on Apple Silicon.

## Architectures

This repository implements **three experimental architectures**, all sharing the same custom GPT-2 backbone and training pipeline:

### Baseline — Vanilla GPT-2

Standard autoregressive transformer. No recursion.

```
Tokens → [Embedding + PosEmb] → Transformer Blocks × N → LayerNorm → LM Head → Logits
```

### Option A — TRM as Post-GPT2 Reasoner

The full GPT-2 stack processes the input, then a TRM pondering block iteratively refines the hidden states before the LM head.

```
Tokens → [GPT-2 all layers] → TRM(z, y, H_cycles, L_cycles) → LM Head → Logits
```

**Hypothesis**: GPT-2 extracts rich syntactic/semantic features; TRM acts as a lightweight "System 2" module that iteratively improves the representation before committing to a vocabulary distribution.

### Option B — Interspersed TRM Layer *(recommended)*

GPT-2 is split at a configurable midpoint. The TRM block operates on the **intermediate hidden states**, which are known to have the highest intrinsic dimensionality and representational capacity.

```
Tokens → [GPT-2 layers 0..mid) → TRM(z, y, H_cycles, L_cycles) → GPT-2 layers mid..N) → LM Head → Logits
```

**Hypothesis**: Placing the recursive reasoning module at the point of maximum representational flexibility should yield the strongest improvements in modeling capacity per added parameter.

## Repository Structure

```
gpt2-text-generation/
├── pyproject.toml                         # uv/pip project config + gpttrm-cli entrypoint
├── data/
│   ├── train_data.csv                     # ~254K training examples (Wikipedia sentences)
│   └── valid_data.csv                     # ~644 validation examples
└── src/gpttrm/
    ├── utils.py                           # Top-k/p sampling, weight loading, truncated normal init
    ├── gpt/
    │   ├── custom_gpt2.py                 # From-scratch GPT-2 with layer-slicing support
    │   ├── trm_block.py                   # TRM reasoning block adapted for continuous hidden states
    │   ├── models.py                      # Lightning modules: GPT2TRMOptionA and GPT2TRMOptionB
    │   ├── train_experiments.py           # Typer CLI for training (gpttrm-cli entrypoint)
    │   ├── gpt2_lm.py                     # Original HuggingFace-based GPT2 LM (reference)
    │   ├── gpt2_tokenizer.py              # GPT-2 tokenizer wrapper
    │   ├── dataloader.py                  # CSV dataset loading
    │   └── training.py                    # Original argparse training script (reference)
    └── trm/
        ├── trm.py                         # Original TRM implementation from the paper
        ├── trm_singlez.py                 # Single-z ablation variant
        ├── trm_hier6.py                   # Hierarchical 6-feature variant
        ├── layers.py                      # Attention, SwiGLU, RoPE, RMSNorm primitives
        ├── losses.py                      # ACT loss head, stablemax cross-entropy
        ├── ema.py                         # Exponential Moving Average helper
        └── sparse_embedding.py            # Sparse puzzle embeddings (original TRM)
```

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- macOS with Apple Silicon (MPS) recommended; CUDA and CPU also supported

### Installation

```bash
# Clone the repository
git clone https://github.com/CarloNicolini/gpttrm.git
cd gpttrm

# Create environment and install all dependencies
uv sync
```

### Quick Start

```bash
# Run baseline GPT-2 for 1 epoch (default)
gpttrm-cli --experiment baseline

# Run Option B (interspersed TRM) with larger batch
gpttrm-cli --experiment option-b --batch-size 64

# Run Option A (post-GPT2 TRM) with custom architecture
gpttrm-cli --experiment option-a --n-embd 512 --n-layer 8 --n-head 8

# Full control over TRM recursion depth
gpttrm-cli --experiment option-b \
    --trm-l-cycles 6 \
    --trm-h-cycles 3 \
    --trm-n-reasoning-layers 2 \
    --trm-insert-layer 3
```

### CLI Reference

| Flag | Description | Default |
|---|---|---|
| `--experiment` | `baseline`, `option-a`, or `option-b` | `baseline` |
| `--batch-size` | Training batch size | `32` |
| `--max-epochs` | Maximum training epochs | `1` |
| `--learning-rate` | AdamW learning rate | `3e-4` |
| `--n-embd` | GPT-2 hidden dimension | `256` |
| `--n-layer` | GPT-2 transformer layers | `6` |
| `--n-head` | GPT-2 attention heads | `4` |
| `--seq-len` | Maximum sequence length | `256` |
| `--trm-l-cycles` | TRM inner recursion steps | `4` |
| `--trm-h-cycles` | TRM outer supervision cycles | `2` |
| `--trm-n-reasoning-layers` | Layers in TRM reasoning module | `2` |
| `--trm-insert-layer` | Option B: insertion point (default: `n_layer//2`) | `None` |

### Monitoring with TensorBoard

```bash
tensorboard --logdir experiments/
```

Each experiment type logs to its own subdirectory (`experiments/baseline/`, `experiments/option-a/`, `experiments/option-b/`) for easy side-by-side comparison.

## Design Decisions

### Custom GPT-2 (not pre-trained)

We deliberately train from random initialization rather than fine-tuning a pre-trained model. This ensures that any performance differences between architectures reflect the structural contribution of the TRM block, not interactions with pre-trained representations.

### Layer Slicing

`CustomGPT2Model.forward()` accepts `start_layer` and `end_layer` arguments, allowing the forward pass to be paused at any layer, the hidden states handed to an external module (the TRM block), and then resumed. This is the key enabler for Option B.

### TRM Adaptations for Language Modeling

The original TRM was designed for fixed-size reasoning tasks (Sudoku, mazes, ARC). Our adaptation makes several changes:

- **No discrete I/O**: The TRM block operates on continuous hidden states from GPT-2, not discrete tokens. Embedding and LM head live in the outer wrapper.
- **No puzzle embeddings**: Removed sparse puzzle embeddings; positional information is already baked into GPT-2's hidden states.
- **Bidirectional attention**: The TRM's internal attention is non-causal, since it refines a complete hidden state rather than generating tokens autoregressively. Causal masking is handled by the surrounding GPT-2 layers.
- **Standard LayerNorm**: Uses GPT-2-style LayerNorm instead of RMSNorm for architectural consistency.

### MPS Optimization

Default hyperparameters are deliberately small (`n_embd=256`, `n_layer=6`, `batch_size=32`, `float32`) to run comfortably on Apple Silicon with ≤16 GB unified memory. Scale up for GPU clusters by increasing these values.

## References

- Martineaux, A. J. (2025). [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871). *arXiv:2510.04871*
- Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). *OpenAI*
- Graves, A. (2016). [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983). *arXiv:1603.08983*
- Aghajanyan, A. et al. (2020). [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255). *arXiv:2012.13255*
- Snell, C. et al. (2024). [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). *arXiv:2408.03314*

## License

MIT
