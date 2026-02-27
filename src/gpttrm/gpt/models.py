"""
GPT-2 + TRM Lightning Modules: Option A and Option B.

Option A – Pre-Head Reasoner:
    Tokens → Full CustomGPT2 → TRM Pondering Block → LM Head

Option B – Interspersed TRM Layer:
    Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from gpttrm.gpt.custom_gpt2 import (
    CustomGPT2Config,
    CustomGPT2Model,
    SharedRecurrentGPT2Model,
    RMSNorm,
)
from gpttrm.gpt.trm_block import TRMBlockConfig, TRMBlock
from gpttrm.gpt.gpt2_tokenizer import GPT2TextEncoder
from gpttrm.gpt.dataloader import text_dataset


# ---------------------------------------------------------------------------
# WSD (Warmup-Stable-Decay) Scheduler Logic
# ---------------------------------------------------------------------------


class WSDScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup-Stable-Decay Scheduler.
    1. Warmup: Linear increase from 0 to peak_lr.
    2. Stable: Constant peak_lr.
    3. Decay: Linear or Cosine decay to min_lr.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear Warmup
            scale = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase
            scale = 1.0
        else:
            # Linear Decay
            decay_steps = self.total_steps - (self.warmup_steps + self.stable_steps)
            current_decay_step = step - (self.warmup_steps + self.stable_steps)
            scale = 1.0 - (current_decay_step / max(1, decay_steps))
            scale = max(0.0, scale)

        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class GPT2TRMBase(pl.LightningModule):
    """Shared boilerplate for both Option A and Option B."""

    def __init__(
        self,
        gpt2_config: CustomGPT2Config,
        trm_config: TRMBlockConfig,
        learning_rate: float = 3e-4,
        lr_scheduler_type: str = "cosine",  # "cosine" or "wsd"
        warmup_steps: int = 2000,
        train_csv: str = "data/train_data.csv",
        dev_csv: str = "data/valid_data.csv",
        test_csv: str = "data/valid_data.csv",
        batch_size: int = 8,
        loader_workers: int = 0,
        total_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gpt2_config", "trm_config"])

        self.gpt2_config = gpt2_config
        self.trm_config = trm_config
        # Ensure TRM respects the modern flag
        self.trm_config.modern = self.gpt2_config.modern

        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.train_csv = train_csv
        self.dev_csv = dev_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.loader_workers = loader_workers

        # Tokenizer (shared across options)
        self.tokenizer = GPT2TextEncoder("gpt2")
        # Update vocab_size to include the added PAD token
        self.gpt2_config.vocab_size = self.tokenizer.vocab_size

        # Core components (to be populated by subclasses)
        self.gpt2: CustomGPT2Model = None  # type: ignore
        self.trm_block: TRMBlock = None  # type: ignore
        self.lm_head: nn.Linear = None  # type: ignore

        # Loss
        self._loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.padding_index)

        # For tracking convergence speed
        self._best_train_loss = float("inf")

    # --- Data ---

    def prepare_sample(self, sample: list) -> dict:
        texts = [s["text"] for s in sample]
        tokens, lengths = self.tokenizer.batch_encode(texts)
        return {"tokens": tokens}

    def train_dataloader(self) -> DataLoader:
        from argparse import Namespace

        hparams = Namespace(
            train_csv=self.train_csv,
            dev_csv=self.dev_csv,
            test_csv=self.test_csv,
        )
        dataset = text_dataset(hparams, val=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.loader_workers,
            shuffle=True,
        )

    # --- Checkpoint Backward Compatibility ---

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Dynamically handles backward compatibility for checkpoints trained
        prior to architectural modernizations.
        """
        state_dict = checkpoint["state_dict"]
        keys = list(state_dict.keys())

        # 1. Translate GPT-2 Attention from combined c_attn to split q_proj, k_proj, v_proj
        for k in keys:
            if k.endswith(".c_attn.weight"):
                v = state_dict.pop(k)  # size is [3 * head_dim * n_head, n_embd]
                q, k_proj, v_proj = v.chunk(3, dim=0)
                prefix = k.replace(".c_attn.weight", "")
                state_dict[prefix + ".q_proj.weight"] = q
                state_dict[prefix + ".k_proj.weight"] = k_proj
                state_dict[prefix + ".v_proj.weight"] = v_proj

                b_key = k.replace(".weight", ".bias")
                if b_key in state_dict:
                    b = state_dict.pop(b_key)
                    qb, kb, vb = b.chunk(3, dim=0)
                    state_dict[prefix + ".q_proj.bias"] = qb
                    state_dict[prefix + ".k_proj.bias"] = kb
                    state_dict[prefix + ".v_proj.bias"] = vb

        # 2. Handle legacy TRMMLP SwiGLU mismatches.
        has_legacy_trm = any("mlp.gate_up_proj.weight" in k for k in keys)
        if has_legacy_trm and not self.trm_config.modern:
            for module in self.modules():
                # Reconstruct TRMMLP to use modern SwiGLU logic dynamically
                if type(module).__name__ == "TRMMLP":
                    module.modern = True
                    hidden_dim = self.trm_config.hidden_size * 4 // 2
                    module.gate_up_proj = nn.Linear(
                        self.trm_config.hidden_size, hidden_dim * 2, bias=False
                    ).to(self.device)
                    module.down_proj = nn.Linear(
                        hidden_dim, self.trm_config.hidden_size, bias=False
                    ).to(self.device)
                    if hasattr(module, "c_fc"):
                        del module.c_fc
                    if hasattr(module, "c_proj"):
                        del module.c_proj
                    if hasattr(module, "gelu"):
                        del module.gelu

    # --- Generation / Inference ---

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generates new tokens autoregressively from a given prompt.
        Args:
            prompt_tokens: (B, T) tensor of input token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = standard, <1.0 = greedy).
            top_k: Top-k sampling filter.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            cond_idx = prompt_tokens[:, -self.gpt2_config.seq_len :]

            result = self(cond_idx)
            # Unpack if model returns (logits, metrics)
            if isinstance(result, tuple):
                logits, _ = result
            else:
                logits = result

            # Pondering models might return very interesting sequences
            # Take logits from the final step and scale
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            prompt_tokens = torch.cat((prompt_tokens, idx_next), dim=1)

        return prompt_tokens

    def val_dataloader(self) -> DataLoader:
        from argparse import Namespace

        hparams = Namespace(
            train_csv=self.train_csv,
            dev_csv=self.dev_csv,
            test_csv=self.test_csv,
        )
        dataset = text_dataset(hparams, train=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.loader_workers,
        )

    # --- Loss & Metrics ---

    def compute_loss(
        self, lm_logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Standard causal LM loss: predict next token."""
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        return self._loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    @torch.no_grad()
    def _compute_token_accuracy(
        self, lm_logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Fraction of next-token predictions that are correct (ignoring padding)."""
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != self.tokenizer.padding_index
        correct = ((preds == shift_labels) & mask).sum()
        total = mask.sum().clamp_min(1)
        return correct.float() / total.float()

    def _log_trm_metrics(self, trm_metrics: dict, prefix: str = "train") -> None:
        """Log TRM-specific metrics to TensorBoard."""
        if trm_metrics is None:
            return
        for key, value in trm_metrics.items():
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=(key in ("trm/y_delta_norm", "trm/z_delta_norm")),
            )

    def _compute_gradient_norms(self) -> dict:
        """Compute gradient norms for GPT-2 and TRM separately."""
        norms = {}
        # GPT-2 gradient norm
        gpt2_grads = [
            p.grad.detach().norm() for p in self.gpt2.parameters() if p.grad is not None
        ]
        if gpt2_grads:
            norms["grad_norm/gpt2"] = torch.stack(gpt2_grads).norm()

        # TRM gradient norm (if TRM block exists)
        if self.trm_block is not None:
            trm_grads = [
                p.grad.detach().norm()
                for p in self.trm_block.parameters()
                if p.grad is not None
            ]
            if trm_grads:
                norms["grad_norm/trm"] = torch.stack(trm_grads).norm()

        # Total gradient norm
        all_grads = [
            p.grad.detach().norm() for p in self.parameters() if p.grad is not None
        ]
        if all_grads:
            norms["grad_norm/total"] = torch.stack(all_grads).norm()

        return norms

    # --- Training / Validation steps ---

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        result = self(tokens)

        # Unpack: TRM models return (logits, trm_metrics), baseline returns just logits
        if isinstance(result, tuple):
            logits, trm_metrics = result
        else:
            logits, trm_metrics = result, {}

        loss = self.compute_loss(logits, tokens)

        # --- Deep Supervision & Halting ACT Loss ---
        deep_sup_logits = trm_metrics.pop("trm/deep_supervision_logits", None)
        halting_logits = trm_metrics.pop("trm/halting_logits", None)

        if deep_sup_logits is not None:
            ds_loss = 0.0
            bce_loss = 0.0

            shift_labels = tokens[..., 1:].contiguous()
            mask = shift_labels != self.tokenizer.padding_index

            # --- Pivot A+B: Weighted Latent-Only Supervision ---
            # We skip supervision on early cycles to allow "pure pondering" (Pivot B)
            # and use increasing weights for later cycles (Pivot A).
            num_cycles = len(deep_sup_logits)

            # Example for 6 cycles: [0, 0, 0.1, 0.2, 0.4, 1.0]
            # This forces the model to specialize Cycle 6 as the "solver" while
            # allowing Cycles 1-3 to build internal representations freely.
            cycle_weights = torch.linspace(0, 1, steps=num_cycles).to(loss.device)
            # Thresholding: Zero out the first 50% of cycles (Latent Reasoning)
            threshold = num_cycles // 2
            cycle_weights[:threshold] = 0.0

            # Normalize weights
            cycle_weights = cycle_weights / (cycle_weights.sum() + 1e-8)

            ds_loss = 0.0
            for h, inter_logits in enumerate(deep_sup_logits):
                w = cycle_weights[h]
                if w > 0:
                    ds_loss += w * self.compute_loss(inter_logits, tokens)

                # 2. ACT Halting BCE
                if halting_logits is not None:
                    # Does the current cycle predict the right token?
                    shift_preds = inter_logits[..., :-1, :].argmax(dim=-1)
                    matches = (shift_preds == shift_labels).float()

                    shift_halting = halting_logits[h][..., :-1, :].squeeze(-1)

                    # Compute BCE only on unmasked (non-pad) tokens
                    bce = F.binary_cross_entropy_with_logits(
                        shift_halting, matches, reduction="none"
                    )
                    bce_loss += (bce * mask).sum() / mask.sum().clamp_min(1)

            # Combined Loss
            # We already averaged the ds_loss via cycle_weights normalization.
            loss = ds_loss

            if halting_logits is not None:
                bce_avg = bce_loss / num_cycles
                loss = loss + 0.1 * bce_avg  # Weight halting loss lightly
                self.log("train/bce_halting", bce_avg, on_step=True)

        # --- Core metrics ---
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Also log as train_loss for backward compat
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        # Perplexity (per-step for training)
        with torch.no_grad():
            ppl = torch.exp(loss.clamp(max=20))  # clamp to avoid inf
            self.log("train/perplexity", ppl, on_step=True, on_epoch=True)

        # Token-level accuracy
        acc = self._compute_token_accuracy(logits, tokens)
        self.log("train/token_accuracy", acc, on_step=True, on_epoch=True)

        # --- TRM-specific metrics ---
        self._log_trm_metrics(trm_metrics, prefix="train")

        # --- Convergence speed tracking ---
        with torch.no_grad():
            if loss.item() < self._best_train_loss:
                self._best_train_loss = loss.item()
            self.log(
                "train/best_loss", self._best_train_loss, on_step=True, on_epoch=False
            )

        return loss

    def on_after_backward(self) -> None:
        """Log gradient norms after backward pass (every N steps to save overhead)."""
        if self.global_step % 50 == 0:
            grad_norms = self._compute_gradient_norms()
            for key, value in grad_norms.items():
                self.log(key, value, on_step=True, on_epoch=False)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        result = self(tokens)

        if isinstance(result, tuple):
            logits, trm_metrics = result
        else:
            logits, trm_metrics = result, None

        loss = self.compute_loss(logits, tokens)

        # --- Core metrics ---
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)  # backward compat

        # Token-level accuracy
        acc = self._compute_token_accuracy(logits, tokens)
        self.log("val/token_accuracy", acc, on_epoch=True)

        # --- TRM-specific metrics ---
        self._log_trm_metrics(trm_metrics, prefix="val")

        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            perplexity = torch.exp(val_loss.clamp(max=20))
            self.log("val/perplexity", perplexity, prog_bar=True)
            self.log("perplexity", perplexity, prog_bar=False)  # backward compat

    def configure_optimizers(self):
        """
        Setup optimizer with selective weight decay.
        Excludes: biases, LayerNorm/RMSNorm weights, and Embeddings.
        """
        # Separate parameters into decay and no_decay groups
        decay = set()
        no_decay = set()

        # Modules that should have weight decay on their weights
        whitelist_weight_modules = (torch.nn.Linear,)
        # Modules that should NOT have weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)

        # We need to track parameters by their identity to handle tied weights
        param_to_name = {p: n for n, p in self.named_parameters()}
        unique_params = list(param_to_name.keys())

        for m in self.modules():
            for pn, p in m.named_parameters():
                if p not in param_to_name:
                    continue

                full_name = param_to_name[p]

                if pn.endswith("bias"):
                    no_decay.add(p)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(p)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(p)
                elif (
                    "z_L_init" in full_name
                    or "z_H_init" in full_name
                    or "gate" in full_name
                ):
                    no_decay.add(p)

        # Catch-all for any missed parameters
        for p in unique_params:
            if p not in decay and p not in no_decay:
                no_decay.add(p)

        # Final check: any parameter in both? (shouldn't happen with set logic but good to be careful)
        decay = decay - no_decay

        optim_groups = [
            {
                "params": sorted(list(decay), key=lambda p: param_to_name[p]),
                "weight_decay": 0.01,
            },
            {
                "params": sorted(list(no_decay), key=lambda p: param_to_name[p]),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.95)
        )

        # --- Multi-Scheduler Support ---
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.total_steps, eta_min=self.learning_rate * 0.1
            )
        elif self.lr_scheduler_type == "wsd":
            # 10% cooldown as per SmolLM recommendation
            warmup = self.warmup_steps
            decay_p = 0.1
            decay_steps = int(self.total_steps * decay_p)
            stable_steps = self.total_steps - warmup - decay_steps

            scheduler = WSDScheduler(
                optimizer,
                warmup_steps=warmup,
                stable_steps=stable_steps,
                total_steps=self.total_steps,
                min_lr=self.learning_rate * 0.1,
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Option A: Pre-Head Reasoner
# ---------------------------------------------------------------------------


class GPT2TRMOptionA(GPT2TRMBase):
    """
    Option A: TRM as a Post-GPT2 Reasoner.

    Flow: Tokens → Full CustomGPT2 (all layers) → TRM Pondering Block → LM Head
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.trm_block = TRMBlock(self.trm_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        hidden_states = self.gpt2(idx=tokens)
        refined, trm_metrics = self.trm_block(hidden_states)
        logits = self.lm_head(refined)
        return logits, trm_metrics


# ---------------------------------------------------------------------------
# Option B: Interspersed TRM Layer
# ---------------------------------------------------------------------------


class GPT2TRMOptionB(GPT2TRMBase):
    """
    Option B: TRM inserted mid-way through GPT-2 layers.

    Flow: Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head
    """

    def __init__(self, trm_insert_layer: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.trm_insert_layer = trm_insert_layer or (self.gpt2_config.n_layer // 2)
        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.trm_block = TRMBlock(self.trm_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # First half of GPT-2
        hidden_states = self.gpt2(
            idx=tokens,
            start_layer=0,
            end_layer=self.trm_insert_layer,
        )
        # TRM pondering at the midpoint
        refined, trm_metrics = self.trm_block(hidden_states)
        # Second half of GPT-2
        hidden_states = self.gpt2(
            inputs_embeds=refined,
            start_layer=self.trm_insert_layer,
            end_layer=self.gpt2_config.n_layer,
        )
        logits = self.lm_head(hidden_states)
        return logits, trm_metrics


# ---------------------------------------------------------------------------
# Option C: The Sandwich (Deep Supervision + Halting)
# ---------------------------------------------------------------------------


class GPT2TRMOptionC(GPT2TRMBase):
    """
    Option C: TRM replaces a block of middle GPT-2 layers (The Sandwich).
    This design utilizes parameter reallocation to increase representational
    depth via TRM without exceeding the original baseline parameter count.

    It supports Truncated BPTT Deep Supervision by parallelizing intermediate
    latent states through the upper network structure.
    """

    def __init__(self, trm_insert_layer: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        # By default, cut out 4 layers from the middle
        self.trm_start_layer = trm_insert_layer or (self.gpt2_config.n_layer // 2 - 2)
        self.trm_end_layer = self.trm_start_layer + 4

        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.trm_block = TRMBlock(self.trm_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # First half of GPT-2 (Perception)
        hidden_states = self.gpt2(
            idx=tokens,
            start_layer=0,
            end_layer=self.trm_start_layer,
        )

        # TRM pondering replacing middle layers
        refined, trm_metrics = self.trm_block(hidden_states)

        intermediates = trm_metrics.pop("trm/intermediates", None)
        halting_logits = trm_metrics.pop("trm/halting_logits", None)

        if self.training and intermediates is not None:
            # --- Deep Supervision Batch Processing ---
            # To train all reasoning cycles (Truncated BPTT), we pass every
            # intermediate reasoning state through the upper portion of GPT-2 at once.
            B, T, D = intermediates[0].shape

            # Stack intermediates along the batch dimension: (H*B, T, D)
            stacked_h = torch.cat(intermediates, dim=0)

            # Parallel evaluation through upper layers (Prediction / Routing)
            stacked_refined = self.gpt2(
                inputs_embeds=stacked_h,
                start_layer=self.trm_end_layer,
                end_layer=self.gpt2_config.n_layer,
            )
            # Evaluate the vocabulary logits for all cycles at once
            all_logits = self.lm_head(stacked_refined)

            # Split back into individual B-sized chunks
            logits_list = list(all_logits.split(B, dim=0))

            # Final output defaults to the very last reasoning cycle
            logits = logits_list[-1]

            # Hand back to training_step to compute Deep Supervision / BCE loss
            trm_metrics["trm/deep_supervision_logits"] = logits_list
            trm_metrics["trm/halting_logits"] = halting_logits
        else:
            # Inference mode: only process the single finalized 'refined' state
            hidden_states = self.gpt2(
                inputs_embeds=refined,
                start_layer=self.trm_end_layer,
                end_layer=self.gpt2_config.n_layer,
            )
            logits = self.lm_head(hidden_states)

        return logits, trm_metrics


# ---------------------------------------------------------------------------
# Shared-Weight Recurrence (TRM-inspired, no TRM machinery)
# ---------------------------------------------------------------------------


class GPT2SharedRecurrent(GPT2TRMBase):
    """
    Shared-weight recurrent GPT-2: the one transferable idea from TRM.

    Architecture:
        Tokens -> bottom_n unique layers -> 1 shared block x K -> top_n unique layers -> LM Head

    Effective depth = bottom_n + K + top_n, parameters ~ bottom_n + 1 + top_n blocks.
    Full gradient flow throughout -- no detach, no dual residual, no deep supervision.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gpt2 = SharedRecurrentGPT2Model(self.gpt2_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.wte.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gpt2(idx=tokens)
        logits = self.lm_head(hidden_states)
        return logits
