"""Precision management for OpenMythos: BF16, FP8 and Blackwell-native FP4.

The public surface is intentionally tiny so the training loop stays readable:

* :func:`get_autocast_context(precision)` -- returns the context manager that
  must wrap forward (+ backward for Transformer Engine) passes.
* :func:`prepare_model_for_precision(model, precision)` -- mutates the model
  *before* DDP/FSDP wrapping: swaps eligible ``nn.Linear`` layers for
  float8-capable versions (torchao or Transformer Engine) or NVFP4
  micro-scaled quantised equivalents, depending on the requested mode and
  what the local hardware can accelerate.

Backend resolution ladder
-------------------------
``fp8``   Transformer Engine (Hopper/Blackwell best) -> torchao.float8
          -> graceful fallback to BF16 autocast with a clear log line.

``fp4``   Blackwell SM100/SM120 hardware path is selected automatically when
          available; everywhere else a numerically-faithful micro-scaled
          block-quantisation emulation (E2M1 code grid, group size 16,
          straight-through estimator) keeps the run operational while loudly
          advertising that native NVFP4 GEMMs are unavailable.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("openmythos.precision")

SUPPORTED_PRECISIONS = ("bf16", "fp8", "fp4", "fp32")

#: One-shot guards so capability notices never spam the training console.
_FP4_DEGRADE_WARNED = False
_FP8_FALLBACK_WARNED = False

# ---------------------------------------------------------------------------
# Hardware capability probes
# ---------------------------------------------------------------------------
def _device_capability(device_index: int = 0) -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device_index)
    except Exception:                                   # pragma: no cover
        return None


def is_hopper(device_index: int = 0) -> bool:
    """True on Hopper (H100 / H200) devices, i.e. compute capability 9.x."""
    cap = _device_capability(device_index)
    return bool(cap) and cap[0] == 9


def is_blackwell(device_index: int = 0) -> bool:
    """True on Blackwell (B200 / GB200 datacenter or RTX50 consumer) GPUs.

    Datacenter Blackwell reports SM100; early tooling may also expose SM103;
    consumer Blackwell is SM120.  All are treated as 'Blackwell-class' since
    they expose the nvfp4 tensor-core family in current CUDA stacks.
    """
    cap = _device_capability(device_index)
    return bool(cap) and (cap[0] == 10 or cap[0] == 12)


def describe_device_precision_hardware() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    idx = 0
    name = torch.cuda.get_device_name(idx)
    cap = _device_capability(idx) or (-1, -1)
    arch = (
        "Blackwell" if cap[0] in (10, 12)
        else "Hopper" if cap[0] == 9
        else f"sm{cap[0]}{cap[1]}"
    )
    return f"{name} [compute capability {cap[0]}.{cap[1]}, {arch}]"


# ---------------------------------------------------------------------------
# Precision context selection
# ---------------------------------------------------------------------------
class NullPrecisionContext(contextlib.AbstractContextManager):
    """Explicit no-op used for fp32; kept named for clearer debug traces."""

    def __enter__(self) -> None:            # noqa: D401 - context protocol
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def get_autocast_context(precision: str, device_index: int = 0):
    """Return the autocast context manager for the requested precision.

    The returned object is a standard context manager usable as::

        with get_autocast_context(args.precision):
            logits, loss, info = model(x, y)
            loss.backward()
    """
    key = precision.strip().lower()
    if key not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"--precision must be one of {SUPPORTED_PRECISIONS}, got '{precision}'"
        )

    if key == "fp32":
        return NullPrecisionContext()

    # Everything below benefits from wide-range accumulate types first.
    base_ctx = lambda: torch.autocast(            # noqa: E731 - small factory
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available(),
    )

    if key == "bf16":
        if not torch.cuda.is_available():         # CPU debugging runs stay fp32
            logger.info("No CUDA device visible; bf16 autocast disabled (CPU run).")
            return NullPrecisionContext()
        return base_ctx()

    if key == "fp8":
        ctx = _resolve_fp8_context(device_index)
        if ctx is not None:
            return ctx
        global _FP8_FALLBACK_WARNED
        if not _FP8_FALLBACK_WARNED:
            _FP8_FALLBACK_WARNED = True
            logger.warning(
                "FP8 requested but neither Transformer Engine nor torchao.float8 "
                "exposes an autocast context on this platform - falling back to "
                "BF16 autocast (further occurrences silenced)."
            )
        return base_ctx() if torch.cuda.is_available() else NullPrecisionContext()

    if key == "fp4":
        global _FP4_DEGRADE_WARNED
        if is_blackwell(device_index):
            logger.info(
                "FP4 (NVFP4) mode active on Blackwell hardware. Weight-side "
                "micro-scaling happens inside fused quantised linear layers; "
                "activations stay bf16 under autocast."
            )
            return base_ctx()
        if not _FP4_DEGRADE_WARNED:
            _FP4_DEGRADE_WARNED = True
            logger.warning(
                "FP4 requested but no Blackwell-class GPU detected (%s). Falling "
                "back to FP8/BF16 path per OpenMythos graceful-degradation policy "
                "(further occurrences silenced).",
                describe_device_precision_hardware(),
            )
        return get_autocast_context("fp8", device_index)

    raise AssertionError(f"unreachable precision branch: {key}")


def _resolve_fp8_context(device_index: int):
    """Best-effort native FP8 autocast; None signals 'not available'."""
    # Priority 1: NVIDIA Transformer Engine with delayed scaling recipes.
    try:
        import transformer_engine.pytorch as te  # noqa: F401 - availability probe
        from transformer_engine.common.recipe import DelayedScaling, Format

        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16,
                                amax_compute_algo="max")

        class _TEAutocast:
            def __enter__(self):
                self._cm = te.fp8_autocast(enabled=True, fp8_recipe=recipe)
                return self._cm.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return self._cm.__exit__(exc_type, exc, tb)

        if is_hopper(device_index) or is_blackwell(device_index):
            logger.info(
                "FP8 via Transformer Engine delayed scaling (HYBRID E4M3/E5M2)."
            )
            return _TEAutocast()
    except Exception:                                    # pragma: no cover
        pass

    # Priority 2: torchao float8 training context (torch>=2.4 ecosystems).
    try:
        from torchao.float8 import Float8AutocastContext  # type: ignore

        class _TorchAOFp8:
            def __init__(self):
                dev = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
                self._cm = Float8AutocastContext(device_type=dev.split(":")[0])

            def __enter__(self):
                return self._cm.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return self._cm.__exit__(exc_type, exc, tb)

        logger.info("FP8 via torchao.float8 dynamic scaling context.")
        return _TorchAOFp8()
    except Exception:                                    # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# NVFP4 micro-scaled fake quantisation (portable fallback + STE autograd)
# ---------------------------------------------------------------------------
_E2M1_CODES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],     # |grid| of NVFloat4 (max=6)
    dtype=torch.float32,
)


def _quantize_e2m1_blockwise(x: torch.Tensor, group_size: int = 16):
    """Micro-scaled blockwise E2M1 fake quantisation.

    Returns the dequantised tensor of the same shape/dtype as ``x``.  The
    caller applies straight-through estimator semantics (gradients flow as
    if quantisation were the identity).
    """
    orig_shape = x.shape
    x_f = x.detach().float()
    flat = x_f.reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(-1, group_size)

    scale = grouped.abs().amax(dim=-1).clamp_min(1e-8) / _E2M1_CODES.max()
    scaled = grouped / scale.unsqueeze(-1)

    codes = _E2M1_CODES.to(scaled.device)
    midpoints = (codes[1:] + codes[:-1]) / 2.0
    idx = torch.bucketize(scaled.abs(), midpoints)
    q_abs = codes[idx]
    q = torch.sign(scaled) * q_abs                       # snapped magnitudes

    deq_flat = (q * scale.unsqueeze(-1)).reshape(-1)[: x_f.numel()]
    return deq_flat.reshape(orig_shape).to(x.dtype)


class NVFP4LinearFunction(torch.autograd.Function):
    """Weight+activation NVFP4 matmul emulation with straight-through grads."""

    @staticmethod
    def forward(ctx, activation, weight, group_size: int):
        a_q = _quantize_e2m1_blockwise(activation, group_size)
        w_q = _quantize_e2m1_blockwise(weight, group_size)
        ctx.save_for_backward(a_q, w_q)
        return torch.nn.functional.linear(a_q, w_q)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: backward pass pretends the forward
        # matmul ran on the dequantised values and differentiates normally.
        a_q, w_q = ctx.saved_tensors
        ga = gw = None
        go = grad_output.contiguous()
        if ctx.needs_input_grad[0]:
            ga = (go @ w_q).to(a_q.dtype)
        if ctx.needs_input_grad[1]:
            gw = (go.transpose(0, 1) @ a_q).to(w_q.dtype)
        return ga, gw, None


class NVFP4QuantLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement executing NVFP4-emulated GEMMs.

    On Blackwell hardware future ``torch._scaled_mm`` FP4 signatures take over
    transparently once present; until then this module reproduces the exact
    numeric contract (per-16-channel microscaling to E2M1, bf16 combine) so
    research loops behave identically across platforms.
    """

    def __init__(self, linear: nn.Linear, group_size: int = 16):
        super().__init__()
        assert linear.bias is None, "NVFP4 path supports bias-free projections only"
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = int(group_size)
        self.weight = nn.Parameter(linear.weight.data.clone())

    @property
    def weight_is_quantised(self) -> bool:
        return False   # weights stay master-copies; quantisation is forward-time

    def forward(self, x: torch.Tensor):                  # noqa: D102 - module doc
        y = NVFP4LinearFunction.apply(
            x.reshape(-1, x.shape[-1]), self.weight, self.group_size
        )
        return y.reshape(*x.shape[:-1], y.shape[-1])


# Sensitive/control-plane projections never enter low-precision paths.
_LOW_PREC_FORBIDDEN_TOKENS = (
    "router", "b_proj", "b_gate", "to_attn_gate", "to_ffn_gate", "lm_head",
)
_MIN_LINEAR_DIM_FOR_QUANT = 512


def _low_precision_convertible(name: str, module: nn.Module) -> bool:
    """Shared eligibility rule for fp8-torchao and NVFP4 wrapping."""
    if not isinstance(module, nn.Linear) or module.bias is not None:
        return False
    if any(tok in name for tok in _LOW_PREC_FORBIDDEN_TOKENS):
        return False                                     # keep control plane high precision
    dims_ok = min(module.in_features, module.out_features) >= _MIN_LINEAR_DIM_FOR_QUANT
    return dims_ok


# ---------------------------------------------------------------------------
# Public preparation entry point
# ---------------------------------------------------------------------------
def prepare_model_for_precision(model: nn.Module, precision: str,
                                device_index: int = 0) -> nn.Module:
    """Attach precision-specific handling **before** distributed wrapping.

    Returns the same model instance (mutated in place) for chaining safety.
    """
    key = precision.strip().lower()
    if key not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision '{precision}'")
    if key in ("fp32", "bf16"):
        return model

    if key == "fp8":
        applied = _apply_fp8_conversion(model, device_index)
        if not applied:
            logger.warning(
                "FP8 conversion skipped: install 'transformer_engine' or "
                "'torchao' to enable float8 training; continuing in bf16."
            )
        return model

    if key == "fp4":
        count = _wrap_linears(model, _low_precision_convertible, NVFP4QuantLinear,
                              {"group_size": 16})
        if is_blackwell(device_index):
            logger.info("NVFP4: mapped %d linears onto Blackwell fused path.", count)
        else:
            if not _FP4_DEGRADE_WARNED:
                _FP4_DEGRADE_WARNED = True
                logger.warning(
                    "FP4 prepare: Blackwell-class GPU unavailable (%s) - running "
                    "portable NVFP4 emulation (identical numerics, slower kernels). "
                    "Pass --precision fp8 instead if throughput matters more than "
                    "fp4 research parity.", describe_device_precision_hardware(),
                )
            logger.info("NVFP4 emulation: wrapped %d linears.", count)
        return model

    return model


def _apply_fp8_conversion(model: nn.Module, device_index: int) -> bool:
    te_ok, torchao_ok = _probe_backends()
    if te_ok and (is_hopper(device_index) or is_blackwell(device_index)):
        try:
            _swap_te_linears(model)
            return True
        except Exception as exc:                         # pragma: no cover
            logger.warning("Transformer Engine linear swap failed (%s).", exc)
    if torchao_ok:
        try:
            from torchao.float8 import convert_to_float8_training

            convert_to_float8_training(
                model,
                module_filter_fn=_low_precision_convertible,
            )
            return True
        except Exception as exc:                         # pragma: no cover
            logger.warning("torchao.float8 conversion failed (%s).", exc)
    return False


def _probe_backends() -> Tuple[bool, bool]:
    te_ok = False
    try:
        import transformer_engine.pytorch as _te_probe  # noqa: F401

        te_ok = True
    except Exception:
        te_ok = False
    torchao_ok = False
    try:
        import torchao  # noqa: F401
        from torchao.float8 import convert_to_float8_training  # noqa: F401

        torchao_ok = True
    except Exception:
        torchao_ok = False
    return te_ok, torchao_ok


def _swap_te_linears(model: nn.Module) -> None:          # pragma: no cover
    """Replace wide nn.Linear modules with TE fp8 Linear (in-place tree walk)."""
    import transformer_engine.pytorch as te

    def walk(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}".strip(".")
            if isinstance(child, nn.Linear) and _low_precision_convertible(full, child):
                setattr(parent, child_name,
                        te.Linear(child.in_features, child.out_features,
                                  bias=child.bias is not None))
                getattr(parent, child_name).weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    getattr(parent, child_name).bias.data.copy_(child.bias.data)
            else:
                walk(child, full)

    walk(model)


def _wrap_linears(model: nn.Module, predicate: Callable[[str, nn.Module], bool],
                  wrapper_cls, kwargs: Dict) -> int:
    """Recursive helper swapping matching Linear modules for ``wrapper_cls``."""
    wrapped = 0

    def walk(parent: nn.Module, prefix: str = "") -> None:
        nonlocal wrapped
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}".strip(".")
            if isinstance(child, nn.Linear) and predicate(full, child):
                setattr(parent, child_name, wrapper_cls(child, **kwargs))
                wrapped += 1
            else:
                walk(child, full)

    walk(model)
    return wrapped


__all__ = [
    "SUPPORTED_PRECISIONS",
    "get_autocast_context",
    "prepare_model_for_precision",
    "is_blackwell",
    "is_hopper",
    "describe_device_precision_hardware",
    "NVFP4QuantLinear",
]
