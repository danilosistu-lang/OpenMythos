"""Sparse Mixture-of-Experts feed-forward networks for OpenMythos.

Implements the DeepSeek-style recipe used by the recurrent core:

* :class:`SharedExpert`  -- a SwiGLU FFN executed unconditionally for every
  token, anchoring common knowledge processing.
* :class:`RoutedExpert`  -- a SwiGLU FFN activated only when the router
  selects it (top-k gating).
* :class:`MythosMoE`     -- the router plus dispatch/combine machinery with
  an auxiliary load-balancing loss, executed as an efficient *batched*
  gather/scatter pass so only ``top_k`` routed experts run per token.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """Standard gated feed-forward block shared by every expert flavour."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        )


class SharedExpert(SwiGLUFFN):
    """SwiGLU expert that runs unconditionally on all tokens.

    The identity gain ``g`` starts at one; routing research (DeepSeek-V2,
    Qwen-MoE) shows a small trainable gain keeps shared vs. routed experts
    from fighting over the residual stream early in training.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__(d_model, hidden_dim, dropout)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * super().forward(x)


class RoutedExpert(SwiGLUFFN):
    """SwiGLU expert activated selectively by the router."""


class MythosMoE(nn.Module):
    """Router + batched top-k MoE layer combining routed and shared experts.

    The forward pass is drop-less: every token is always processed by its
    chosen ``top_k`` experts regardless of per-expert capacity, and the
    auxiliary load-balancing loss keeps assignments from collapsing.

    After each forward the differentiable scalar :attr:`last_aux_loss`
    holds this layer's load-balancing penalty; the causal LM head sums it
    across layers into the training objective.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        hidden_dim: int,
        dropout: float = 0.0,
        renormalize: bool = True,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k
        self.renormalize = renormalize

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.routed_experts = nn.ModuleList(
            [RoutedExpert(d_model, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SharedExpert(d_model, hidden_dim, dropout) for _ in range(num_shared_experts)]
        )

        #: Populated after every forward; lives on the autograd graph.
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_router_entropy: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ gates
    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute top-k routing decisions.

        Returns:
            weights   ``[N, k]``   normalised combine weights per token/expert.
            indices   ``[N, k]``   flat expert ids selected per token.
            aux_loss  scalar tensor: switch-style load balancing penalty.
        """
        logits = self.router(x).float()                    # fp32 softmax stability
        probs = torch.softmax(logits, dim=-1)              # [N, E]

        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        if self.renormalize:
            denom = top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            top_p = top_p / denom

        n_tok, n_exp = probs.shape
        # Switch-style balance loss: alpha * E * sum_e f_e * P_e where f_e is
        # the fraction of routed slots assigned to expert e and P_e the mean
        # router probability mass it received. Minimised at uniform usage.
        with torch.autocast(device_type=x.device.type, enabled=False):
            assignment_one_hot = F.one_hot(top_i, num_classes=n_exp).to(logits.dtype)
            f = assignment_one_hot.mean(dim=(0, 1))        # [E] slot fraction
            p_mean = probs.mean(dim=0)                     # [E] prob mass
            aux = n_exp * torch.sum(f * p_mean)

        self.last_router_entropy = float(-(probs.log() * probs).sum(-1).mean().detach())
        return top_p.to(x.dtype), top_i, aux

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, S, D]`` or ``[N, D]``; output matches input shape."""
        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])                  # [N, D]
        n_tok = flat.shape[0]

        weights, idx, aux = self._route(flat)
        self.last_aux_loss = aux

        out = torch.zeros_like(flat)

        # ---- unconditional shared experts ---------------------------------
        shared_sum = None
        for se in self.shared_experts:
            term = se(flat)
            shared_sum = term if shared_sum is None else shared_sum + term

        # ---- batched routed-expert dispatch --------------------------------
        flat_idx = idx.reshape(-1)                         # [N*k]
        flat_w = weights.reshape(-1)                       # [N*k]
        tok_rows = (
            torch.arange(n_tok, device=x.device)
            .unsqueeze(-1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        sort_order = torch.argsort(flat_idx)               # group slots per expert
        sorted_experts = flat_idx[sort_order]
        sorted_rows = tok_rows[sort_order]
        sorted_weights = flat_w[sort_order]

        boundaries = torch.searchsorted(
            sorted_experts,
            torch.arange(self.num_experts + 1, device=x.device, dtype=sorted_experts.dtype),
        )
        boundaries = boundaries.tolist()

        for e in range(self.num_experts):
            lo, hi = boundaries[e], boundaries[e + 1]
            if lo == hi:
                continue
            rows = sorted_rows[lo:hi]
            expert_in = flat.index_select(0, rows)
            expert_out = self.routed_experts[e](expert_in)
            contributions = expert_out * sorted_weights[lo:hi].unsqueeze(-1).to(expert_out.dtype)
            out.index_add_(0, rows, contributions.to(out.dtype))

        if shared_sum is not None:
            out = out + shared_sum

        return out.reshape(*leading_shape, -1)


__all__ = ["MythosMoE", "SharedExpert", "RoutedExpert", "SwiGLUFFN"]
