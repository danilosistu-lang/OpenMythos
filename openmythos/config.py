"""Configuration system for the OpenMythos Recurrent-Depth Transformer.

A :class:`MythosConfig` fully describes every architectural hyper-parameter of
an OpenMythos model.  Factory presets exist for seven reference sizes, ranging
from ``100m`` (edge / single-GPU experiments) up to ``10b`` (multi-node
Blackwell/Hopper fleets).  The presets scale width, head layout, expert pool
sizes and loop depth together so every variant trains stably with one recipe.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict


def _round_to(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return int(math.ceil(value / multiple) * multiple)


def swiglu_hidden_dim(d_model: int, multiplier: float, multiple_of: int = 128) -> int:
    """SwiGLU inner width used by dense FFNs and MoE experts alike.

    Follows the standard Llama-style heuristic (``8/3 * d_model`` scaled by
    ``multiplier``) rounded up so tensor-core friendly tile sizes are hit.
    """
    return _round_to(int(float(d_model) * multiplier * (2.0 / 3.0)), multiple_of)


ATTN_TYPES = ("gqa", "mla")


@dataclass
class MythosConfig:
    """Full architecture specification of an OpenMythos RDT model."""

    # Registry of reference presets (filled below, see ``_MODEL_VARIANTS``).
    _VARIANTS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------ core
    model_variant: str = "custom"
    vocab_size: int = 50257                # GPT-2 byte-level BPE vocabulary
    d_model: int = 1024
    n_heads: int = 16                      # query heads (GQA & MLA)
    n_kv_heads: int = 4                    # grouped key/value heads (GQA only)

    # ------------------------------------------------------- stage structure
    prelude_layers: int = 2                # encode tokens -> latent state 'e'
    recurrent_layers: int = 3              # weight-shared base blocks inside loop
    coda_layers: int = 2                   # decode latent state -> predictions

    # ------------------------------------------------------------ recurrence
    max_loop_iters: int = 8                # T: recurrent block executions
    dropout: float = 0.0

    # ------------------------------------------------------------- attention
    attn_type: str = "gqa"                 # "gqa" | "mla"
    rope_head_dim: int = 64                # RoPE lanes (MLA decoupled position part)
    qk_nope_head_dim: int = 0              # 0 => auto-derive (= min(d/n_heads, 128))
    v_head_dim: int = 0                    # 0 => auto-derive (= qk_nope_head_dim)
    rope_theta: float = 10000.0
    use_flash_attn: bool = True            # prefer FlashAttention-2/3 when importable
    attn_softmax_scale: float = 0.0        # 0 => 1/sqrt(effective head dim)

    # ------------------------------------------------------------------- MLA
    kv_lora_rank: int = 512                # compressed KV latent rank (DeepSeek-style)
    q_lora_rank: int = 0                   # 0 => auto-derive (= min(2*d, 1536))

    # ------------------------------------------------------------------- MoE
    num_experts: int = 8                   # number of *routed* experts
    num_shared_experts: int = 1            # always-on experts (unconditional)
    top_k_experts: int = 2                 # routed experts activated per token
    aux_loss_coeff: float = 0.01           # load-balancing auxiliary loss weight
    z_loss_coeff: float = 1e-3             # router logit-magnitude (ST-MoE z) penalty

    # ------------------------------------------------------------------- FFN
    ffn_multiplier: float = 8.0 / 3.0      # SwiGLU width heuristic multiplier
    ffn_multiple_of: int = 128             # alignment granularity of inner width

    # ---------------------------------------------------- DepthWise adapters
    lora_rank: int = 16                    # rank r of per-loop-step adapters

    # ------------------------------------------------------------- sequences
    max_seq_len: int = 4096

    # ------------------------------------------------------------------ misc
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True            # share tok-embedding with lm_head
    gradient_checkpointing: bool = False   # recompute loop steps to save VRAM
    bos_token_id: int = 50256
    eos_token_id: int = 50256

    # Extra metadata (kept for forward compatibility / experiment bookkeeping).
    notes: Dict[str, Any] = None  # type: ignore[assignment]

    # ================================================================= checks
    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = {}
        if self.attn_type not in ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {ATTN_TYPES}, got '{self.attn_type}'"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for Grouped-Query Attention"
            )
        if not 1 <= self.top_k_experts <= self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be within "
                f"[1, num_experts={self.num_experts}]"
            )
        if min(self.prelude_layers, self.recurrent_layers, self.coda_layers) < 1:
            raise ValueError(
                "all stages need at least one layer "
                "(prelude_layers, recurrent_layers, coda_layers)"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        # Derived attention geometry --------------------------------------
        if self.qk_nope_head_dim <= 0:
            self.qk_nope_head_dim = min(self.head_dim, 128)
        if self.v_head_dim <= 0:
            self.v_head_dim = self.qk_nope_head_dim
        if self.q_lora_rank <= 0:
            self.q_lora_rank = min(2 * self.d_model, 1536)
        self.gqa_group_size = self.n_heads // self.n_kv_heads
        self.effective_gqa_head_dim = self.head_dim
        self.mla_q_head_dim = self.qk_nope_head_dim + self.rope_head_dim

    # ========================================================== shape helpers
    @property
    def head_dim(self) -> int:
        """Query-head dimensionality used by Grouped-Query Attention."""
        return self.d_model // self.n_heads

    @property
    def expert_hidden_dim(self) -> int:
        """Inner width of every SwiGLU expert feed-forward network."""
        return swiglu_hidden_dim(self.d_model, self.ffn_multiplier, self.ffn_multiple_of)

    @property
    def effective_depth(self) -> int:
        """Total transformer-layer traversals per token at inference time."""
        return (
            self.prelude_layers
            + self.recurrent_layers * self.max_loop_iters
            + self.coda_layers
        )

    def gqa_scale(self) -> float:
        if self.attn_softmax_scale > 0.0:
            return self.attn_softmax_scale
        return 1.0 / math.sqrt(self.effective_gqa_head_dim)

    def mla_scale(self) -> float:
        if self.attn_softmax_scale > 0.0:
            return self.attn_softmax_scale
        return 1.0 / math.sqrt(self.mla_q_head_dim)

    # ============================================================== factories
    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation (stored inside checkpoints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MythosConfig":
        """Rebuild a config from :meth:`to_dict` output (checkpoint resume)."""
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in known}
        cfg = cls(**filtered)
        cfg.__dict__.update({k: v for k, v in payload.items() if k not in known})
        return cfg

    @classmethod
    def from_variant(cls, variant_str: str, **overrides: Any) -> "MythosConfig":
        """Instantiate a preset configuration for a named model size.

        Accepts forms such as ``"1b"``, ``"500m"``, ``"10B"`` or bare digits
        like ``"700"`` (interpreted as millons).  Keyword overrides win over
        preset values, e.g.
        ``MythosConfig.from_variant("1b", attn_type="mla", max_loop_iters=12)``.
        """
        key = variant_str.strip().lower()
        if key.isdigit():
            key = f"{key}m"
        if key not in cls._VARIANTS:
            raise KeyError(
                f"Unknown model variant '{variant_str}'. "
                f"Known variants: {sorted(cls._VARIANTS)}"
            )
        payload: Dict[str, Any] = {"model_variant": key}
        payload.update(cls._VARIANTS[key])
        payload.update(overrides)
        return cls(**payload)

    def describe(self) -> str:
        """Human-readable multi-line summary used for startup logging."""
        return (
            f"OpenMythos[{self.model_variant}] d_model={self.d_model} "
            f"heads={self.n_heads} kv_heads={self.n_kv_heads}\n"
            f"  stage : prelude x{self.prelude_layers} -> loop x{self.max_loop_iters} "
            f"(stack of {self.recurrent_layers} shared blocks) -> coda x{self.coda_layers}\n"
            f"  attn  : {self.attn_type.upper()} flash={self.use_flash_attn}\n"
            f"  moe   : routed={self.num_experts}(+{self.num_shared_experts} shared) "
            f"top-{self.top_k_experts} lora_r={self.lora_rank}\n"
            f"  seq={self.max_seq_len} vocab={self.vocab_size} "
            f"effective_depth={self.effective_depth}"
        )


#: Reference presets consumed by :meth:`MythosConfig.from_variant`.
#:
#: Sizing philosophy: the *total* parameter count (every routed expert
#: counted) lands inside the advertised bucket, while the always-executed
#: cost per token stays a small fraction of the total thanks to top-k
#: routing and the weight-shared recurrent stack -- stacking more base
#: blocks multiplies the expert pool, so OpenMythos prefers additional loop
#: iterations ``T`` over additional stacked blocks wherever possible.
_MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "100m": dict(
        d_model=640, n_heads=10, n_kv_heads=2,
        prelude_layers=1, recurrent_layers=2, coda_layers=1,
        num_experts=12, top_k_experts=2,
        max_seq_len=2048, tie_embeddings=True, ffn_multiplier=8.0 / 3.0,
    ),
    "300m": dict(
        d_model=768, n_heads=12, n_kv_heads=3,
        prelude_layers=2, recurrent_layers=3, coda_layers=2,
        num_experts=24, top_k_experts=2,
        max_seq_len=2048, tie_embeddings=True, ffn_multiplier=8.0 / 3.0,
    ),
    "500m": dict(
        d_model=1024, n_heads=16, n_kv_heads=4,
        prelude_layers=2, recurrent_layers=2, coda_layers=2,
        num_experts=32, top_k_experts=2,
        max_seq_len=4096, tie_embeddings=True, ffn_multiplier=8.0 / 3.0,
    ),
    "1b": dict(
        d_model=1536, n_heads=24, n_kv_heads=6,
        prelude_layers=2, recurrent_layers=2, coda_layers=2,
        num_experts=24, top_k_experts=2,
        max_seq_len=4096, tie_embeddings=False, ffn_multiplier=8.0 / 3.0,
    ),
    "3b": dict(
        d_model=2560, n_heads=40, n_kv_heads=10,
        prelude_layers=2, recurrent_layers=2, coda_layers=2,
        num_experts=32, top_k_experts=2,
        max_seq_len=4096, tie_embeddings=False, ffn_multiplier=8.0 / 3.0,
    ),
    "7b": dict(
        d_model=3840, n_heads=30, n_kv_heads=10,
        prelude_layers=2, recurrent_layers=2, coda_layers=2,
        num_experts=40, top_k_experts=2,
        max_seq_len=4096, tie_embeddings=False, ffn_multiplier=8.0 / 3.0,
    ),
    "10b": dict(
        d_model=4480, n_heads=35, n_kv_heads=7,
        prelude_layers=2, recurrent_layers=2, coda_layers=2,
        num_experts=40, top_k_experts=2,
        max_seq_len=4096, tie_embeddings=False, ffn_multiplier=8.0 / 3.0,
    ),
}
MythosConfig._VARIANTS.update(_MODEL_VARIANTS)

__all__ = [
    "MythosConfig",
    "KNOWN_VARIANTS",
    "ATTN_TYPES",
    "swiglu_hidden_dim",
]

KNOWN_VARIANTS = tuple(_MODEL_VARIANTS.keys())
