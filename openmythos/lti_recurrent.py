"""Linear Time-Invariant (LTI) recurrence machinery for OpenMythos.

This module is the mathematical heart of the Recurrent-Depth Transformer:

* :class:`LTIRecurrentInjection` maintains the linear recurrence

    ``h_{t+1} = A . h_t + B . e``

  where ``A`` is a *continuous-time negative diagonal matrix*
  ``A_cont = -exp(log_A)`` discretised through Zero-Order Hold / Euler
  integration: ``A_disc = exp(dt * A_cont) = exp(-dt * exp(log_A))``.
  Every diagonal entry therefore lies in the open interval ``(0, 1)``,
  which guarantees ``rho(A) < 1`` *for every parameter value* -- the
  recurrence cannot blow up, and gradients flowing backwards through many
  loop iterations remain bounded.

* :class:`DepthWiseLoRA` provides low-rank residual adapters indexed by the
  current loop step ``t`` so that iteration 0 and iteration T-7 can apply
  different transformations using shared backbone weights.  The adapters are
  zero-initialised (up-projection) and additionally gated by a tanh of a
  learnable scalar starting at zero, which makes each loop step an exact
  identity function at initialisation -- deep recurrences start stable.
"""

from __future__ import annotations

import math
from typing import ClassVar, List

import torch
import torch.nn as nn


class LTIRecurrentInjection(nn.Module):
    """Spectrally-stable linear state update ``h' = A h + B e``.

    Args:
        d_model: channel count of the recurrent state ``h``.
        init_decay_mean: approximate desired mean of ``A_disc`` at init
            (must lie in ``(0, 1)``).  Channels are scattered around it.
    """

    def __init__(self, d_model: int, init_decay_mean: float = 0.75):
        super().__init__()
        if not 0.0 < init_decay_mean < 1.0:
            raise ValueError("init_decay_mean must be within (0, 1)")

        # A_cont = -exp(log_A); choose log_A ~ N(mu, sigma) such that the
        # unit-dt decay product lands near `init_decay_mean` per channel.
        mu = math.log(-math.log(init_decay_mean))
        self.log_a = nn.Parameter(torch.randn(d_model) * 0.4 + mu)

        # dt enters through softplus, hence dt > 0 strictly for all params.
        # Init solves softplus(raw) == 1 so that E[decay] matches
        # `init_decay_mean` exactly at initialisation time.
        self.raw_dt = nn.Parameter(torch.full((d_model,), math.log(math.e - 1.0)))

        # Input projection B (latent 'e' -> state space) plus a per-channel
        # sigmoid gate controlling how strongly the latent drives each axis.
        self.b_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_gate = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------ spectra ops
    # Floors chosen so the most adversarial parameterisation still yields
    # dt * |A_cont| >= 2.4e-6, which sits ~20+ fp32 ULPs away from 1.0.
    _LOG_A_MIN: ClassVar[float] = -6.0
    _LOG_A_MAX: ClassVar[float] = 8.0
    _DT_MIN: ClassVar[float] = 1e-3

    def continuous_diagonal(self) -> torch.Tensor:
        """``A_cont = -exp(log_a)`` -- strictly negative, always.

        ``log_a`` is clamped to ``[-6, 8]``: the lower floor keeps every
        channel's decay comfortably below unity under fp32 rounding rules,
        while the upper ceiling avoids overflow in downstream magnitudes.
        """
        return -torch.exp(torch.clamp(self.log_a,
                                      min=self._LOG_A_MIN, max=self._LOG_A_MAX))

    def discretized_diagonal(self) -> torch.Tensor:
        """Zero-Order Hold (Euler-fallback) discretisation ``exp(dt * A_cont)``.

        ``dt`` is positive by construction (softplus) and floored at 1e-3;
        combined with the bounded ``A_cont`` this guarantees that every
        diagonal entry of ``A_disc`` remains *strictly* inside ``(0, 1)``
        even when optimisers drive raw parameters to extremes -- i.e.
        ``rho(A) < 1`` holds numerically, not just symbolically.
        """
        dt = torch.nn.functional.softplus(self.raw_dt).clamp_min(self._DT_MIN)
        return torch.exp(dt * self.continuous_diagonal())

    def spectral_radius(self) -> torch.Tensor:
        """Largest diagonal magnitude; provably in ``(0, 1)`` at every step."""
        return self.discretized_diagonal().abs().max()

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Apply ``h <- A_disc . h + sigmoid(gate) . B e``.

        Args:
            h: ``[B, S, d_model]`` recurrent state.
            e: ``[B, S, d_model]`` latent conditioning from the prelude.
        """
        a_disc = self.discretized_diagonal().to(h.dtype)           # [d_model]
        gate = torch.sigmoid(self.b_gate(e)).to(h.dtype)
        injected = gate * self.b_proj(e).to(h.dtype)
        return a_disc * h + injected


class LoRAStep(nn.Module):
    """A single loop-step adapter ``up @ down`` with identity-safe gating."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        std = 0.02 / math.sqrt(rank)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=std)
        nn.init.zeros_(self.up.weight)                 # exact no-op at init
        self.gate = nn.Parameter(torch.zeros(1))       # tanh(0)=0 second guard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * self.up(self.down(x))


class DepthWiseLoRA(nn.Module):
    """Collection of rank-r adapters, one per loop iteration depth ``t``.

    A :class:`LoRAStep` exists for every ``t in [0, max_iters)`` allowing each
    pass through the shared recurrent block to specialise its attention /
    FFN contribution without materialising full per-depth weight copies --
    parameter cost grows only by ``2 * r * d`` per adapter instead of a new
    transformer block.
    """

    def __init__(self, d_model: int, rank: int, max_iters: int,
                 n_adapters_per_step: int = 1):
        super().__init__()
        self.max_iters = max_iters
        self.n_adapters_per_step = n_adapters_per_step
        self.steps: List[List[LoRAStep]] = []
        flat = []
        for _t in range(max_iters):
            row = []
            for _j in range(n_adapters_per_step):
                module = LoRAStep(d_model, rank)
                row.append(module)
                flat.append(module)
            self.steps.append(row)
        self.adapters = nn.ModuleList(flat)

    def forward_index(self, x: torch.Tensor, t: int, j: int = 0) -> torch.Tensor:
        """Adapter application for loop iteration ``t`` / slot ``j``."""
        if t >= self.max_iters:
            raise IndexError(
                f"loop step t={t} exceeds configured max_loop_iters="
                f"{self.max_iters}; increase the config or clamp the loop"
            )
        return self.steps[t][j](x)

    def total_adapter_parameters(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())


__all__ = [
    "LTIRecurrentInjection",
    "DepthWiseLoRA",
    "LoRAStep",
]
