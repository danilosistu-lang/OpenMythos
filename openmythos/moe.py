"""OpenMythos Recurrent-Depth Transformer (RDT) model definition.

Three-stage architecture (per the OpenMythos specification):

1. **Prelude**  -- ordinary transformer blocks run once, projecting token
   embeddings into a continuous latent state representation ``e``.
2. **Recurrent core** -- a weight-shared stack of ``recurrent_layers`` blocks
   executed ``max_loop_iters`` times.  Each loop step blends the state with
   the LTI injection ``h <- A h + B e``, then applies the shared attention +
   MoE block whose outputs are specialised by per-step DepthWise LoRA
   adapters::

       h_{t+1} = A_disc * h_t + B(e) + LoRA_t(Block(h_t))

3. **Coda** -- final dense blocks plus RMSNorm feeding the vocabulary head.

Supports per-loop-step gradient checkpointing to trade compute for VRAM on
large variants, weight tying for small variants, and both GQA and MLA
attention backbones selected purely through :class:`MythosConfig`.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    GQAttention,
    MLAttention,
    RMSNorm,
    build_rope_tables,
    flash_attention_backend,
)
from .config import MythosConfig
from .lti_recurrent import DepthWiseLoRA, LTIRecurrentInjection
from .moe import MythosMoE, SwiGLUFFN


def make_attention(cfg: MythosConfig) -> nn.Module:
    """Instantiate the configured attention backbone."""
    if cfg.attn_type == "gqa":
        return GQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            dropout=cfg.dropout,
            use_flash_attn=cfg.use_flash_attn,
            scale=cfg.gqa_scale(),
        )
    if cfg.attn_type == "mla":
        return MLAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            dropout=cfg.dropout,
            use_flash_attn=cfg.use_flash_attn,
            scale=cfg.mla_scale(),
        )
    raise ValueError(f"Unsupported attn_type '{cfg.attn_type}'")


class DenseTransformerBlock(nn.Module):
    """Dense block used by the prelude / coda stages (non-MoE)."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = make_attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.expert_hidden_dim, cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class LatentConditioner(nn.Module):
    """Projects latent ``e`` into multiplicative gates that modulate each
    sub-layer input inside the recurrent block.

    Feature-wise modulation lets every loop iteration reinterpret the *same*
    frozen input differently depending on where it sits relative to the
    latent conditioning signal, without inflating the backbone parameters.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=1e-6)
        self.to_attn_gate = nn.Linear(d_model, d_model, bias=False)
        self.to_ffn_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        basis = self.norm(e)
        # (1 + tanh(.)) keeps gates near unity and prevents sign flipping of
        # the residual stream early in training.
        g_attn = 1.0 + torch.tanh(self.to_attn_gate(basis))
        g_ffn = 1.0 + torch.tanh(self.to_ffn_gate(basis))
        return g_attn, g_ffn


class RecurrentBlock(nn.Module):
    """Weight-shared block executed at every loop step ``t``.

    Combines (in order): latent conditioning -> grouped/latent attention ->
    depth-wise LoRA -> sparse MoE FFN -> depth-wise LoRA -> LTI injection.
    All adaptive behaviour is indexed by ``(t)``; all heavy weights are
    shared across iterations by construction.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.conditioner = LatentConditioner(cfg.d_model)

        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = make_attention(cfg)
        self.lora_attn = DepthWiseLoRA(
            cfg.d_model, cfg.lora_rank, cfg.max_loop_iters, n_adapters_per_step=1
        )

        self.moe_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.moe = MythosMoE(
            d_model=cfg.d_model,
            num_experts=cfg.num_experts,
            num_shared_experts=cfg.num_shared_experts,
            top_k=cfg.top_k_experts,
            hidden_dim=cfg.expert_hidden_dim,
            dropout=cfg.dropout,
        )
        self.lora_moe = DepthWiseLoRA(
            cfg.d_model, cfg.lora_rank, cfg.max_loop_iters, n_adapters_per_step=1
        )

        self.lti = LTIRecurrentInjection(cfg.d_model)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        t: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one weight-shared loop step; returns ``(h_next, aux, z)``.

        The two routing losses are *returned* rather than read from module
        attributes after the fact: under gradient checkpointing the first
        (no-grad) pass would otherwise overwrite the attributes with
        graph-less tensors, silently detaching the load-balancing and
        z-loss objectives from the optimiser -- a classic recipe for
        router drift and transient loss spikes.
        """
        g_attn, g_ffn = self.conditioner(e)

        # --- attention sub-layer with depth adapter -------------------------
        u = self.attn(self.attn_norm(h) * g_attn, cos, sin)
        h = h + self.resid_dropout(self.lora_attn.forward_index(u, t))

        # --- MoE feed-forward sub-layer with depth adapter -------------------
        m = self.moe(self.moe_norm(h) * g_ffn)
        h = h + self.resid_dropout(self.lora_moe.forward_index(m, t))

        # --- spectrally-stable LTI state injection --------------------------
        h = self.lti(h, e)
        return h, self.moe.last_aux_loss, self.moe.last_z_loss


class PreludeStack(nn.Module):
    """Stage 1: encode tokens into states and produce latent reference ``e``."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DenseTransformerBlock(cfg) for _ in range(cfg.prelude_layers)]
        )
        self.latent_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, cos, sin)
        e = self.latent_norm(x)
        return x, e


class CodaStack(nn.Module):
    """Stage 3: post-processing layers before the vocabulary projection."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DenseTransformerBlock(cfg) for _ in range(cfg.coda_layers)]
        )

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, cos, sin)
        return h


# ===========================================================================
# Full causal LM
# ===========================================================================
class OpenMythosForCausalLM(nn.Module):
    """The complete Recurrent-Depth Transformer language model."""

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.prelude = PreludeStack(config)
        self.recurrent_stack = nn.ModuleList(
            [RecurrentBlock(config) for _ in range(config.recurrent_layers)]
        )
        self.coda = CodaStack(config)
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        if config.tie_embeddings:
            self.lm_head = None      # weight tying resolved in forward/projection
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        rope_len = max(config.max_seq_len, 8192) + 1
        cos, sin = build_rope_tables(rope_len, _rope_width(config), config.rope_theta,
                                     device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.max_loop_iters: int = int(config.max_loop_iters)
        self.gradient_checkpointing: bool = bool(config.gradient_checkpointing)

        #: Diagnostics filled by the last forward pass (kept as plain floats).
        self.last_ce_loss: float = float("nan")
        self.last_aux_loss: float = float("nan")
        self.last_z_loss: float = float("nan")
        self.last_loop_iters: int = self.max_loop_iters

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------------- internals
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Standard ~1/sqrt(effective_depth) shrink on residual write-out
        projections (attention ``wo`` and expert/FFN ``w_down``), keeping
        activation magnitudes stable across very deep unrolled loops."""
        depth = max(self.config.effective_depth, 4)
        gain = 1.0 / math.sqrt(depth)
        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                with torch.no_grad():
                    param.mul_(gain)

    def projection_weight(self) -> torch.Tensor:
        """Return the (possibly tied) unembedding matrix."""
        if self.lm_head is not None:
            return self.lm_head.weight
        return self.token_embedding.weight

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loop_iters: Optional[int] = None,
    ):
        """Run the three-stage RDT flow.

        Args:
            idx:        ``[B, S]`` int64 token ids.
            targets:    ``[B, S]`` int64 next-token ids (shifted by caller).
                        When omitted no loss is computed (pure inference).
            loop_iters: optional runtime override of the recurrent depth T;
                        defaults to ``config.max_loop_iters``.  Enables
                        train-short / test-deep style adaptation.
        """
        bsz, seq_len = idx.shape
        if seq_len > self.rope_cos.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds cached RoPE horizon "
                f"{self.rope_cos.shape[0]} - raise MythosConfig.max_seq_len"
            )

        device = idx.device
        cos = self.rope_cos[:seq_len].to(device=device, dtype=torch.float32)
        sin = self.rope_sin[:seq_len].to(device=device, dtype=torch.float32)

        x = self.token_embedding(idx)
        x = self.embedding_dropout(x)

        # ---- stage 1: prelude ----------------------------------------------
        h, e = self.prelude(x, cos, sin)

        # ---- stage 2: recurrent core ----------------------------------------
        # aux/z routing losses are collected per *execution* (the stack is
        # weight-shared, so every loop iteration re-routes) and returned
        # through the block -- this keeps them attached to the autograd
        # graph even when every execution runs under grad checkpointing.
        steps = loop_iters if loop_iters is not None else self.max_loop_iters
        aux_terms: List[torch.Tensor] = []
        z_terms: List[torch.Tensor] = []
        for t in range(steps):
            for block in self.recurrent_stack:
                if self.gradient_checkpointing and self.training:
                    step_fn = partial(block, t=t, cos=cos, sin=sin)
                    out = torch.utils.checkpoint.checkpoint(
                        step_fn, h, e, use_reentrant=False
                    )
                else:
                    out = block(h, e, t=t, cos=cos, sin=sin)
                h = out[0]
                if len(out) > 1 and out[1] is not None:
                    aux_terms.append(out[1])
                    z_terms.append(out[2])

        # ---- stage 3: coda + projection --------------------------------------
        out = self.final_norm(self.coda(h, cos, sin))
        logits = F.linear(out, self.projection_weight())

        loss_dict = {"ce": 0.0, "aux": 0.0, "z": 0.0}
        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
            self.last_ce_loss = float(ce.detach())
            self.last_aux_loss = (
                sum(float(a.detach()) for a in aux_terms) / max(len(aux_terms), 1))
            self.last_z_loss = (
                sum(float(z_.detach()) for z_ in z_terms) / max(len(z_terms), 1))
            self.last_loop_iters = int(steps)

            # Total objective: CE + balance loss + router z-loss.  Both
            # auxiliary terms are stacked from live, graph-attached tensors
            # (they survived checkpointing by construction above).
            total = ce
            if aux_terms:
                total = total + self.config.aux_loss_coeff * torch.stack(aux_terms).mean()
            if z_terms:
                total = total + self.config.z_loss_coeff * torch.stack(z_terms).mean()
            loss_dict["ce"] = self.last_ce_loss
            loss_dict["aux"] = self.last_aux_loss
            loss_dict["z"] = self.last_z_loss
            loss = total

        return logits, loss, loss_dict

    # ------------------------------------------------------------ MoE helpers
    def iter_recurrent_blocks(self) -> Iterator[RecurrentBlock]:
        for blk in self.recurrent_stack:
            yield blk

    def iter_moe_layers(self) -> Iterator[MythosMoE]:
        for blk in self.recurrent_stack:
            yield blk.moe

    def aux_loss_tensor(self) -> torch.Tensor:
        """Legacy helper: mean balance loss from the last forward's attrs.

        The training forward no longer relies on this (it consumes the
        per-execution terms returned by :class:`RecurrentBlock`, which
        survive gradient checkpointing); kept for external callers.
        """
        terms = [
            moe.last_aux_loss
            for moe in self.iter_moe_layers()
            if moe.last_aux_loss is not None and isinstance(moe.last_aux_loss, torch.Tensor)
        ]
        if not terms:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(terms).mean()

    # ----------------------------------------------------------- capabilities
    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        """Toggle recompute-over-reuse on loop steps (VRAM saver)."""
        self.gradient_checkpointing = bool(enabled)
        self.config.gradient_checkpointing = bool(enabled)

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> torch.optim.Optimizer:
        """AdamW with dimension-aware decay grouping (fused when possible)."""
        decay_params, nodecay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (decay_params if param.ndim >= 2 else nodecay_params).append(param)
        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        on_cuda = next(self.parameters()).device.type == "cuda"
        try:
            return torch.optim.AdamW(
                groups, lr=lr, betas=betas, eps=eps, fused=on_cuda
            )
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def _rope_width(cfg: MythosConfig) -> int:
    """RoPE lane width depends on the active attention backbone."""
    if cfg.attn_type == "gqa":
        return cfg.head_dim
    return cfg.rope_head_dim


__all__ = [
    "OpenMythosForCausalLM",
    "PreludeStack",
    "RecurrentBlock",
    "CodaStack",
    "DenseTransformerBlock",
    "flash_attention_backend",
]
