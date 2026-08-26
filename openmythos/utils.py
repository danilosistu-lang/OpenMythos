"""Training utilities for OpenMythos.

Contents
--------
* :func:`count_parameters` -- total vs. *active-per-token* parameter accounting
  that understands MoE routing fractions (only ``top_k`` routed experts plus
  the always-on shared experts actually execute for every token).
* :func:`estimate_flops_per_token` -- approximate training FLOPs used for
  MFU reporting.
* :class:`CosineAnnealingWithWarmup` / :func:`get_lr` -- LR schedule.
* :func:`setup_distributed`, :func:`wrap_model` -- DDP / FSDP bootstrapping
  driven by GPU count and model size.
* :class:`LoggerWrapper` -- unified console / WandB / TensorBoard logging.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import MythosConfig
from .moe import MythosMoE

logger = logging.getLogger("openmythos.utils")


# ===========================================================================
# Parameter accounting
# ===========================================================================
def _module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_parameters(model: nn.Module, config: Optional[MythosConfig] = None
                     ) -> Dict[str, float]:
    """Parameter census differentiating MoE totals from active per-token cost.

    Returns a dict with keys::

        total_parameters              all weights incl. idle routed experts
        active_parameters_per_token   parameters traversed by one token:
                                      everything except (1 - top_k/E) of the
                                      routed-expert matrices
        routed_expert_parameters      aggregate size of the expert pool
        shared_expert_parameters      always-on experts + router + control
    """
    config = config or getattr(model, "config", None)

    total = sum(p.numel() for p in model.parameters())
    routed = 0
    shared = 0
    n_experts = n_shared = top_k = 0

    for module in model.modules():
        if isinstance(module, MythosMoE):
            n_experts = max(n_experts, module.num_experts)
            n_shared = max(n_shared, len(module.shared_experts))
            top_k = max(top_k, module.top_k)
            routed += sum(_module_param_count(e) for e in module.routed_experts)
            shared += sum(_module_param_count(e) for e in module.shared_experts)
            shared += _module_param_count(module.router)

    # Every recurrent block *shares* one pool; modules() iterates unique
    # parameters exactly once, so `routed` above already reflects the shared
    # stack correctly (no double counting across loop iterations).
    if n_experts and routed > 0:
        activation_fraction = min(top_k / n_experts, 1.0)
        active_routed = int(round(routed * activation_fraction))
    else:
        active_routed = 0

    active = total - routed + active_routed

    result: Dict[str, float] = {
        "total_parameters": total,
        "active_parameters_per_token": active,
        "routed_expert_parameters": routed,
        "shared_expert_parameters": shared,
        "non_expert_parameters": total - routed - shared,
    }
    if config is not None:
        result["effective_depth"] = config.effective_depth
    return result


def estimate_flops_per_token(config: MythosConfig, seq_len: Optional[int] = None,
                             include_attention: bool = True) -> float:
    """Back-of-envelope *forward* FLOPs per token.

    Uses the classic ``~2 x params`` contract-layer estimate plus quadratic
    attention terms; training cost is roughly three times this figure once
    backward/recompute work is included.  Useful for MFU reporting, not for
    hardware procurement decisions.
    """
    d = config.d_model
    ff = config.expert_hidden_dim

    def macs_dense(n_layers: int) -> float:
        # attention projections (wq, wk, wv, wo) + SwiGLU FFN (gate, up, down)
        return n_layers * (4.0 * d * d + 3.0 * ff * d)

    if config.attn_type == "mla":
        q_compress = d * config.q_lora_rank + (
            config.q_lora_rank * config.n_heads * config.mla_q_head_dim
        )
        kv_compress = d * (config.kv_lora_rank + config.rope_head_dim) + (
            config.kv_lora_rank
            * config.n_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
        )
        attention_proj_macs = 4.0 * d * d - 2.0 * d * d + q_compress + kv_compress \
            + d * config.n_heads * config.v_head_dim   # wo
    else:
        attention_proj_macs = 4.0 * d * d               # wq/wk/wv/wo

    moe_active_macs = (
        config.top_k_experts + config.num_shared_experts
    ) * 3.0 * ff * d                                    # gate/up/down
    router_macs = d * config.num_experts
    loop_reps = max(config.max_loop_iters, 1)

    proj_macs = (
        macs_dense(config.prelude_layers)
        + loop_reps * config.recurrent_layers * (
            attention_proj_macs + moe_active_macs + router_macs
        )
        + macs_dense(config.coda_layers)
    )
    total = 2.0 * proj_macs                             # 2 MAC->FLOP
    total += 2.0 * d * config.vocab_size                # unembedding

    if include_attention and seq_len:
        effective_layers = (
            config.prelude_layers
            + loop_reps * config.recurrent_layers
            + config.coda_layers
        )
        score_and_value = 4.0 * seq_len * config.n_heads * config.head_dim
        extra = 8.0 * seq_len * config.head_dim if config.attn_type == "mla" else 0.0
        total += effective_layers * (score_and_value + extra)
    return total


# ===========================================================================
# Learning-rate scheduling
# ===========================================================================
def get_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int,
           min_lr_ratio: float = 0.1) -> float:
    """Linear warmup followed by cosine decay to ``base_lr * min_ratio``."""
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_term


@dataclass
class CosineAnnealingWithWarmup:
    """Callable scheduler: linear warmup then cosine decay, manual stepping."""

    optimizer: torch.optim.Optimizer
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        self._last_step: int = 0

    @property
    def last_step(self) -> int:
        return self._last_step

    def current_lr(self) -> float:
        return get_lr(self._last_step, self.base_lr, self.warmup_steps,
                      self.total_steps, self.min_lr_ratio)

    def __call__(self, step: int) -> float:
        lr = get_lr(step, self.base_lr, self.warmup_steps,
                    self.total_steps, self.min_lr_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_step = step
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.base_lr = sd["base_lr"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.min_lr_ratio = sd["min_lr_ratio"]

    _step: int = 0


# ===========================================================================
# Distributed helpers
# ===========================================================================
@dataclass
class DistState:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    backend: str = "gloo"
    is_main: bool = True


def setup_distributed() -> DistState:
    """Initialise torch.distributed when launched via ``torchrun``/srun.

    Returns an inert single-process state otherwise.  Safe to call multiple
    times; subsequent calls reuse the already-initialised view.
    """
    state = DistState()
    env_world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    state.rank = int(os.environ.get("RANK", 0))
    state.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    state.world_size = max(env_world, 1)

    if torch.cuda.is_available():
        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        n_visible = torch.cuda.device_count()
        if state.world_size > n_visible and env_world > 1:
            logger.warning(
                "WORLD_SIZE=%d but only %d CUDA devices visible.",
                state.world_size, n_visible,
            )
        state.device = f"cuda:{state.local_rank % max(n_visible, 1)}"
    else:
        backend = "gloo"
        state.device = "cpu"
    state.backend = backend

    if state.world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend, world_size=state.world_size, rank=state.rank,
            timeout=timedelta(minutes=30),
        )
        torch.distributed.barrier()
    state.is_main = state.rank == 0
    if state.is_main:
        logger.info(
            "Distributed init complete: rank=%d/%d backend=%s device=%s",
            state.rank, state.world_size, backend, state.device,
        )
    return state


def teardown_distributed() -> None:
    with contextlib.suppress(Exception):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


_MODEL_SIZE_FSDP_THRESHOLD_BYTES = 8_000_000_000      # ~8B params * fp16/2 bytes heuristic


def wrap_model(model: nn.Module, dist: DistState, strategy: str = "auto",
               gradient_checkpointing_hint: bool = False) -> nn.Module:
    """Wrap ``model`` for distributed data-parallel execution.

    ``strategy`` accepts ``'auto' | 'ddp' | 'fsdp' | 'none'``.  Under
    ``auto``, models at or above the large-variant threshold run under
    Fully Sharded Data Parallel (with bf16 mixed precision and transformer
    layer auto-wrapping); smaller ones use plain DDP.
    """
    if strategy not in ("auto", "ddp", "fsdp", "none"):
        raise ValueError(f"unknown wrap strategy '{strategy}'")
    if strategy == "none" or dist.world_size == 1:
        return model.to(dist.device)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    wants_fsdp = strategy == "fsdp" or (
        strategy == "auto" and param_bytes >= _MODEL_SIZE_FSDP_THRESHOLD_BYTES // 2
    )

    if wants_fsdp:
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                BackwardPrefetch,
                CPUOffload,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import (
                ModuleWrapPolicy,
            )

            wrap_targets = []
            try:
                from .model import DenseTransformerBlock, RecurrentBlock
                wrap_targets = [RecurrentBlock, DenseTransformerBlock]
            except ImportError:                        # pragma: no cover
                wrap_targets = []

            policy = (
                ModuleWrapPolicy(wrap_targets) if wrap_targets else None
            )
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16 if dist.backend == "nccl" else None,
                reduce_dtype=torch.bfloat16 if dist.backend == "nccl" else None,
                buffer_dtype=torch.bfloat16 if dist.backend == "nccl" else None,
            )
            return FSDP(
                model.to(dist.device),
                auto_wrap_policy=policy,
                mixed_precision=mp_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=False),
                device_id=dist.local_rank % max(torch.cuda.device_count(), 1),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                limit_all_gathers=True,
                use_orig_params=True,
                sync_module_states=True,   # broadcast rank-0 weights (seeded identically)
            )
        except Exception as exc:                       # pragma: no cover
            logger.warning("FSDP wrapping failed (%s); falling back to DDP.", exc)

    find_unused = any(isinstance(m, MythosMoE) for m in model.modules())
    ddp_model = nn.parallel.DistributedDataParallel(
        model.to(dist.device),
        device_ids=[dist.local_rank] if dist.device.startswith("cuda") else None,
        find_unused_parameters=find_unused,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    return ddp_model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the pristine nn.Module beneath DDP/FSDP wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


# ===========================================================================
# Logging
# ===========================================================================
@dataclass
class LoggerWrapper:
    """Console-first logger with optional WandB and TensorBoard fan-out."""

    project: str = "openmythos-fineweb"
    run_name: str = field(default_factory=lambda: f"openmythos-{datetime.now():%Y%m%d-%H%M%S}")
    config_payload: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tensorboard_dir: str = "./tb_logs"

    wandb_run: Any = None
    tb_writer: Any = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.project, name=self.run_name, config=self.config_payload,
                mode=os.environ.get("WANDB_MODE", "online") or "online",
                reinit="finish_previous",
            )
        except Exception as exc:                        # pragma: no cover
            logger.info("wandb unavailable (%s); using TensorBoard/console.", exc)
            self.wandb_run = None
        if self.wandb_run is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                os.makedirs(self.tensorboard_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)
            except Exception as exc:                    # pragma: no cover
                logger.info("TensorBoard unavailable (%s); console-only.", exc)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pretty = " | ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in metrics.items())
        logger.info("[%d] %s", step, pretty)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        elif self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, global_step=step)
            self.tb_writer.flush()

    def finish(self) -> None:
        if self.wandb_run is not None:
            with contextlib.suppress(Exception):
                self.wandb_run.finish()
        if self.tb_writer is not None:
            with contextlib.suppress(Exception):
                self.tb_writer.close()


def set_seed_everywhere(seed: int, deterministic_cudnn: bool = False) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:                                 # pragma: no cover
        pass
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = not deterministic_cudnn


__all__ = [
    "count_parameters",
    "estimate_flops_per_token",
    "get_lr",
    "CosineAnnealingWithWarmup",
    "DistState",
    "setup_distributed",
    "teardown_distributed",
    "wrap_model",
    "unwrap_model",
    "LoggerWrapper",
    "set_seed_everywhere",
]
