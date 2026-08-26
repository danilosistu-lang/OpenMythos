"""OpenMythos: a production-grade Recurrent-Depth Transformer (RDT).

Public surface keeps heavy optional imports (flash-attn, transformer_engine,
torchao, datasets, tiktoken) strictly lazy so ``import openmythos`` stays
fast and dependency-free.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "MythosConfig",
    "OpenMythosForCausalLM",
    "count_parameters",
]


def __getattr__(name: str):  # lazy re-exports keep startup cost near zero
    if name == "MythosConfig":
        from .config import MythosConfig

        return MythosConfig
    if name == "OpenMythosForCausalLM":
        from .model import OpenMythosForCausalLM

        return OpenMythosForCausalLM
    if name == "count_parameters":
        from .utils import count_parameters

        return count_parameters
    raise AttributeError(f"module 'openmythos' has no attribute '{name}'")
