"""High-performance attention primitives for OpenMythos.

Two interchangeable attention backbones are provided:

* :class:`GQAttention` -- Grouped-Query Attention with Rotary Position
  Embeddings (RoPE).  Execution is dispatched to the fastest available kernel:
  FlashAttention-3 (Blackwell/Hopper), FlashAttention-2, or the native
  ``torch.nn.functional.scaled_dot_product_attention`` (Flash / Mem-Efficient
  back-ends compiled into PyTorch 2.4+).

* :class:`MLAttention` -- DeepSeek-style Multi-Latent Attention.  Keys and
  values are compressed into a shared low-rank latent vector which is what
  would be cached at inference time; position information is carried by a
  decoupled RoPE lane that bypasses the compression.

Both modules expose the same ``forward(x, cos, sin)`` contract so the model
body can switch between them purely from configuration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FlashAttention backend discovery (FA-3 -> FA-2 -> SDPA fallback chain)
# ---------------------------------------------------------------------------
_FLASH_FUNC = None          # Callable or None
_FLASH_BACKEND = "sdpa"     # one of: "fa3", "fa2", "sdpa"

try:  # FlashAttention-3 ships inside flash-attn>=2.7 for Hopper; standalone on Blackwell.
    from flash_attn_interface import flash_attn_func as _FLASH_FUNC  # type: ignore

    _FLASH_BACKEND = "fa3"
except Exception:  # pragma: no cover - depends on optional wheel presence
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _FLASH_FUNC  # type: ignore

        _FLASH_BACKEND = "fa2"
    except Exception:
        _FLASH_FUNC = None
        _FLASH_BACKEND = "sdpa"


def flash_attention_backend() -> str:
    """Name of the concrete attention kernel actually selected at import."""
    return _FLASH_BACKEND


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split style rotary embedding application.

    Args:
        x:   ``[B, S, H, D]`` activations whose last dimension is even.
        cos: ``[S, D/2]`` or broadcastable cosine table.
        sin: ``[S, D/2]`` or broadcastable sine table.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    if cos.dim() == 2:                      # [S, D/2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def build_rope_tables(seq_len: int, head_dim: int, theta: float,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute ``cos``/``sin`` tables of shape ``[seq_len, head_dim // 2]``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [S, D/2]
    return freqs.cos(), freqs.sin()


class GQAttention(nn.Module):
    """Grouped-Query Attention with RoPE and fused-kernel dispatch."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, dropout: float = 0.0,
                 use_flash_attn: bool = True, scale: float = 0.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "GQA requires n_heads % n_kv_heads == 0"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_flash_attn = bool(use_flash_attn)
        self.scale = scale if scale > 0 else 1.0 / math.sqrt(head_dim)

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """``[B, Hkv, S, D] -> [B, Hkv*n_rep, S, D]``."""
        if n_rep == 1:
            return x
        bsz, kv_heads, seq, dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq, dim)
        return x.reshape(bsz, kv_heads * n_rep, seq, dim)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                causal: bool = True) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        flash_ok = (
            _FLASH_FUNC is not None
            and self.use_flash_attn
            and x.is_cuda
            and seq_len > 1
            and causal
            and x.dtype in (torch.bfloat16, torch.float16)
        )

        if flash_ok:
            y = _FLASH_FUNC(
                q.contiguous(),                       # [B, S, H, D]
                k.contiguous(),
                v.contiguous(),
                causal=True,
            )
            if isinstance(y, tuple):                  # FA-3 returns (out, lse)
                y = y[0]
            y = y.to(x.dtype)
        else:
            qh = q.transpose(1, 2)                    # [B, H, S, D]
            kh = self._repeat_kv(k.transpose(1, 2), self.n_rep)
            vh = self._repeat_kv(v.transpose(1, 2), self.n_rep)
            y = F.scaled_dot_product_attention(
                qh, kh, vh,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=causal and seq_len > 1,
                scale=self.scale,
            )                                          # [B, H, S, D]
            y = y.transpose(1, 2)                      # [B, S, H, D]

        out = self.wo(y.reshape(bsz, seq_len, self.n_heads * self.head_dim))
        return out


class MLAttention(nn.Module):
    """DeepSeek-style Multi-Latent Attention with compressed KV latents.

    Layout per head:

    * ``q_nope``  : content query lanes (``qk_nope_head_dim``)
    * ``q_pe``    : RoPE query lanes      (``rope_head_dim``, decoupled)
    * latent      : shared compressed KV vector of width ``kv_lora_rank``
    * keys/values : uncompressed on the fly from the latent via ``wkv_b``
    """

    def __init__(self, d_model: int, n_heads: int, q_lora_rank: int,
                 kv_lora_rank: int, qk_nope_head_dim: int, rope_head_dim: int,
                 v_head_dim: int, dropout: float = 0.0,
                 use_flash_attn: bool = True, scale: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + rope_head_dim
        self.use_flash_attn = bool(use_flash_attn)
        self.scale = scale if scale > 0 else 1.0 / math.sqrt(self.q_head_dim)

        # Query path ---------------------------------------------------------
        self.wq_a = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank, eps=1e-6)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)

        # Fused key-latent + decoupled rope-key projection --------------------
        self.wk_a = nn.Linear(d_model, kv_lora_rank + rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps=1e-6)
        self.wkv_b = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        self.wo = nn.Linear(n_heads * v_head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        #: Compressed latent produced by the most recent forward pass.  At
        #: inference time this tensor (plus the small ``k_pe`` rope key) is all
        #: that must be cached - hence the name *multi-latent* attention.
        self.last_latent: Optional[torch.Tensor] = None
        self.last_k_pe: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                causal: bool = True) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h, d_rk = self.n_heads, self.qk_nope_head_dim

        # ---- queries ------------------------------------------------------
        cq = self.wq_a(x)
        q = self.wq_b(self.q_norm(cq))
        q = q.view(bsz, seq_len, h, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [d_rk, self.rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, cos, sin)

        # ---- compressed KV latent -----------------------------------------
        ck_kp = self.wk_a(x)                                   # [B, S, rank + rope]
        c_kv, k_pe_raw = torch.split(
            ck_kp, [self.kv_lora_rank, self.rope_head_dim], dim=-1
        )
        c_kv = self.kv_norm(c_kv)
        k_pe = apply_rope(
            k_pe_raw.view(bsz, seq_len, 1, self.rope_head_dim), cos, sin
        )
        # retain references for inference-time cache plumbing ---------------
        self.last_latent = c_kv.detach() if not torch.is_grad_enabled() else c_kv
        self.last_k_pe = k_pe.detach() if not torch.is_grad_enabled() else k_pe

        kv = self.wkv_b(c_kv)                                  # [B, S, H, dk+dv]
        kv = kv.view(bsz, seq_len, h, d_rk + self.v_head_dim)
        k_nope, v = torch.split(kv, [d_rk, self.v_head_dim], dim=-1)

        # ---- explicit score computation -------------------------------------
        # NOTE: unlike GQA, MLA mixes heterogeneous head widths (q/k lanes of
        # qk_nope+rope versus v lanes of v_head_dim), so fused FA kernels only
        # apply to the absorbed variant; the training-time math below is the
        # reference (non-absorbed) formulation and remains numerically exact.
        k_pe_full = k_pe.expand(bsz, seq_len, h, self.rope_head_dim)
        scores = torch.einsum("bshd,bthd->bhst", q_nope, k_nope)
        scores = scores + torch.einsum("bshd,bthd->bhst", q_pe, k_pe_full)
        scores = scores * self.scale                           # [B, H, S_q, S_k]

        if causal and seq_len > 1:
            mask = torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~mask, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        probs = self.attn_dropout(probs)

        ctx = torch.einsum("bhst,bthd->bshd", probs, v)        # [B, S, H, dv]
        out = self.wo(ctx.reshape(bsz, seq_len, h * self.v_head_dim))
        return out


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation used across OpenMythos."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float()
        norm = norm * torch.rsqrt(norm.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(x.dtype)


__all__ = [
    "GQAttention",
    "MLAttention",
    "RMSNorm",
    "apply_rope",
    "build_rope_tables",
    "flash_attention_backend",
]
