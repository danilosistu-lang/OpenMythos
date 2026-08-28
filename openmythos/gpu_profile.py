"""GPU auto-detection and MFU-oriented tuning profiles for OpenMythos.

This module answers one question: *given the silicon actually inside this
box, which training settings maximise Model FLOPs Utilisation?*  It ships a
curated database of NVIDIA GPUs spanning every architecture from Pascal
(GTX 10xx / P100) through Volta, Turing (T4), Ampere, Ada, Hopper and both
Blackwell generations (datacenter B100/B200/B300 and consumer RTX 50xx),
derives a tuned setting bundle per card, and can apply it to a training run.

Typical use from the training entrypoint::

    python train.py --auto_tune ...

or standalone inspection / experimentation::

    python scripts/tune_gpu.py                 # pretty report for this host
    python scripts/tune_gpu.py --simulate B300 # what would a B300 get?
    python scripts/tune_gpu.py --self-test     # validate the whole DB

Design notes
------------
* **Torch-free detection.**  The primary probe shells out to ``nvidia-smi``
  so ``--print`` / ``--json`` work on hosts without PyTorch installed.
* **Honest peaks.**  Peak numbers are *dense* (no sparsity marketing) and
  approximate; they exist to turn the trainer's MFU gauge from a hardcoded
  H100 constant into a per-card figure.  Marked ``est`` when synthesised.
* **Conservative by default.**  The tuner never invents exotic kernels: it
  picks among settings ``train.py`` already supports (precision, flash
  attention on/off, batch/accum split, checkpointing, worker counts) and
  only *advises* on opt-in extras such as FP8 Transformer Engine.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("openmythos.gpu")

# ===========================================================================
# Architecture families: what each generation can actually accelerate
# ===========================================================================
FAMILIES: Dict[str, Dict[str, Any]] = {
    # tensor-core fp16 / bf16 / tf32 / fp8 / fp4  +  flash-attn  +  compile
    "pascal":       dict(tc_fp16=False, bf16=False, tf32=False, fp8=False,
                         fp4=False, flash=False, compile=False,
                         sdp_flash=False, sdp_cudnn=False,
                         label="Pascal (no tensor cores)"),
    "volta":        dict(tc_fp16=True, bf16=False, tf32=False, fp8=False,
                         fp4=False, flash=False, compile=False,
                         sdp_flash=False, sdp_cudnn=False,
                         label="Volta (fp16 tensor cores)"),
    "turing":       dict(tc_fp16=True, bf16=False, tf32=False, fp8=False,
                         fp4=False, flash=False, compile=True,
                         sdp_flash=False, sdp_cudnn=False,
                         label="Turing (fp16 tensor cores)"),
    "ampere":       dict(tc_fp16=True, bf16=True, tf32=True, fp8=False,
                         fp4=False, flash=True, compile=True,
                         sdp_flash=True, sdp_cudnn=True,
                         label="Ampere"),
    "ada":          dict(tc_fp16=True, bf16=True, tf32=True, fp8=True,
                         fp4=False, flash=True, compile=True,
                         sdp_flash=True, sdp_cudnn=True,
                         label="Ada Lovelace"),
    "hopper":       dict(tc_fp16=True, bf16=True, tf32=True, fp8=True,
                         fp4=False, flash=True, compile=True,
                         sdp_flash=True, sdp_cudnn=True,
                         label="Hopper"),
    "bw_dc":        dict(tc_fp16=True, bf16=True, tf32=True, fp8=True,
                         fp4=True, flash=False, compile=True,
                         sdp_flash=False, sdp_cudnn=True,
                         label="Blackwell datacenter"),
    "bw_consumer":  dict(tc_fp16=True, bf16=True, tf32=True, fp8=True,
                         fp4=True, flash=True, compile=True,
                         sdp_flash=True, sdp_cudnn=True,
                         label="Blackwell consumer"),
}

#: dense TFLOPS used when an unknown card must be synthesised from CC tier
_PEAKS_PER_SM = {          # (tf32, bf16, fp16, fp8, fp4) dense TF per SM
    (6, 0): (0.0, 0.0, 0.0, 0.0, 0.0),
    (7, 0): (0.0, 0.0, 1.6, 0.0, 0.0),      # V100: 125 / 80
    (7, 5): (0.0, 0.0, 1.6, 0.0, 0.0),      # T4:   65 / 40
    (8, 0): (1.4, 2.9, 2.9, 0.0, 0.0),      # A100: 312 / 108
    (8, 6): (0.9, 0.9, 0.9, 0.0, 0.0),      # 3090: 71  / 82
    (8, 7): (1.5, 3.0, 3.0, 0.0, 0.0),
    (8, 9): (1.1, 1.3, 1.3, 2.6, 0.0),      # 4090: 165 / 128
    (9, 0): (3.7, 7.5, 7.5, 15.0, 0.0),     # H100: 990 / 132
    (10, 0): (7.4, 15.2, 15.2, 30.4, 60.8),  # B200: 2250 / 148
    (10, 3): (6.2, 12.5, 12.5, 25.0, 93.8),  # B300 fp4-lean
    (12, 0): (0.6, 1.2, 1.2, 2.5, 9.9),     # 5090: 210 / 170
}

_CUDA_MIN_BY_CC = {(10, 0): (12, 8), (10, 3): (12, 8),
                   (12, 0): (12, 8), (12, 1): (12, 8)}

# ===========================================================================
# GPU database (peaks are DENSE, approximate, no-sparsity marketing numbers)
# ===========================================================================
def _g(name: str, alias: List[str], cc: Tuple[int, int], family: str,
       sms: int, vram: float, bw: int, tdp: int,
       fp32: float = 0.0, tf32: float = 0.0, bf16: float = 0.0,
       fp16: float = 0.0, fp8: float = 0.0, fp4: float = 0.0,
       tc: Optional[bool] = None, note: str = "") -> Dict[str, Any]:
    return dict(name=name, alias=alias, cc=cc, family=family, sms=sms,
                vram=vram, bw=bw, tdp=tdp,
                peaks=dict(fp32=fp32, tf32=tf32, bf16=bf16, fp16=fp16,
                           fp8=fp8, fp4=fp4),
                tc=(FAMILIES[family]["tc_fp16"] if tc is None else tc),
                note=note)


GPU_DB: List[Dict[str, Any]] = [
    # ------------------------------------------------------- Pascal (6.x)
    _g("Tesla P100", ["P100-PCIE", "P100 SXM"], (6, 0), "pascal", 56, 16,
       732, 300, fp32=21.2, fp16=42.4, tc=False,
       note="fp16 runs at 2x via CUDA cores, no tensor cores"),
    _g("Tesla P40", ["P40"], (6, 1), "pascal", 60, 24, 347, 250,
       fp32=12.2, note="no fp16 acceleration at all"),
    _g("Tesla P4", ["P4 "], (6, 1), "pascal", 20, 8, 192, 75,
       fp32=5.7, note="low-power inference card"),
    # -------------------------------------------------------- Volta (7.0)
    _g("Tesla V100", ["V100-SXM", "V100-PCIE", "V100-32GB", "V100 16GB"],
       (7, 0), "volta", 80, 32, 900, 300, fp32=15.7, fp16=125,
       note="use fp16+GradScaler: real 125 TF tensor cores"),
    # ------------------------------------------------------- Turing (7.5)
    _g("Tesla T4", ["T4"], (7, 5), "turing", 40, 14.5, 320, 70,
       fp32=8.1, fp16=65,
       note="70 W cap; fp16 tensor cores are the only win. vram=USABLE "
            "14.5 GiB (16 GB nominal; clouds report 14.56 total capacity)"),
    _g("RTX 2080 Ti", ["2080 Ti"], (7, 5), "turing", 68, 11, 616, 250,
       fp32=13.4, fp16=107.6),
    _g("RTX 2080", ["RTX 2080"], (7, 5), "turing", 46, 8, 448, 215,
       fp32=10.1, fp16=80.8),
    _g("RTX 2070", ["RTX 2070"], (7, 5), "turing", 36, 8, 448, 175,
       fp32=7.8, fp16=62.4),
    _g("GTX 1660 Ti", ["1660 Ti"], (7, 5), "turing", 24, 6, 288, 120,
       fp32=5.4, fp16=5.4, tc=False,
       note="GTX 16xx: Turing WITHOUT tensor cores (fp16 runs at fp32 rate)"),
    _g("GTX 1650", ["GTX 1650"], (7, 5), "turing", 16, 4, 128, 75,
       fp32=3.0, fp16=3.0, tc=False,
       note="GTX 16xx: no tensor cores"),
    # ---------------------------------------------- Ampere datacenter (8.0)
    _g("A100 80GB", ["A100-SXM4-80GB", "A100 80GB", "A100-SXM 80"],
       (8, 0), "ampere", 108, 80, 2039, 400, fp32=19.5, tf32=156, bf16=312,
       fp16=312, note="NVLink; the reference trainer card"),
    _g("A100 40GB", ["A100-SXM4-40GB", "A100 PCIE", "A100 40GB"],
       (8, 0), "ampere", 108, 40, 1555, 300, fp32=19.5, tf32=156, bf16=312,
       fp16=312),
    _g("A800 80GB", ["A800"], (8, 0), "ampere", 108, 80, 2039, 400,
       fp32=19.5, tf32=156, bf16=312, fp16=312,
       note="A100 derivative (NVLink capped at 400 GB/s)"),
    _g("A30", ["A30"], (8, 0), "ampere", 56, 24, 933, 165,
       fp32=9.7, tf32=82.5, bf16=165, fp16=165),
    # ------------------------------------------------ Ampere rest (8.6/8.7)
    _g("A10G", ["A10G"], (8, 6), "ampere", 80, 24, 600, 150,
       fp32=31.5, tf32=63, bf16=125, fp16=125),
    _g("A10", ["A10"], (8, 6), "ampere", 72, 24, 600, 150,
       fp32=31.2, tf32=62.5, bf16=125, fp16=125),
    _g("A40", ["A40"], (8, 6), "ampere", 107, 48, 696, 300,
       fp32=37.4, tf32=74.8, bf16=149.6, fp16=149.6),
    _g("RTX A6000", ["A6000"], (8, 6), "ampere", 84, 48, 768, 300,
       fp32=38.7, tf32=77.4, bf16=154.8, fp16=154.8),
    _g("RTX A5000", ["A5000"], (8, 6), "ampere", 64, 24, 768, 230,
       fp32=27.5, tf32=54.2, bf16=108.4, fp16=108.4),
    _g("RTX A4000", ["A4000"], (8, 6), "ampere", 48, 16, 448, 140,
       fp32=19.2, tf32=38.4, bf16=76.8, fp16=76.8),
    _g("A2", ["A2"], (8, 6), "ampere", 44, 16, 200, 60,
       fp32=4.5, tf32=9, bf16=18, fp16=18, note="edge server card"),
    _g("Jetson AGX Orin 64GB", ["Orin"], (8, 7), "ampere", 16, 64, 205, 60,
       fp32=5.3, tf32=10.6, bf16=42.5, fp16=42.5,
       note="unified memory; keep num_workers small"),
    # ------------------------------------------------- Ampere consumer
    _g("RTX 3090 Ti", ["3090 Ti"], (8, 6), "ampere", 84, 24, 1008, 450,
       fp32=40, tf32=80, bf16=80, fp16=80),
    _g("RTX 3090", ["RTX 3090"], (8, 6), "ampere", 82, 24, 936, 350,
       fp32=35.6, tf32=71, bf16=71, fp16=71,
       note="no NVLink P2P on multi-GPU rigs"),
    _g("RTX 3080 Ti", ["3080 Ti"], (8, 6), "ampere", 68, 12, 760, 350,
       fp32=34.1, tf32=68.2, bf16=68.2, fp16=68.2),
    _g("RTX 3080", ["RTX 3080"], (8, 6), "ampere", 68, 10, 760, 320,
       fp32=29.8, tf32=59.5, bf16=59.5, fp16=59.5),
    _g("RTX 3070 Ti", ["3070 Ti"], (8, 6), "ampere", 48, 8, 608, 290,
       fp32=21.8, tf32=43.5, bf16=43.5, fp16=43.5),
    _g("RTX 3070", ["RTX 3070"], (8, 6), "ampere", 46, 8, 448, 220,
       fp32=20.3, tf32=40.6, bf16=40.6, fp16=40.6),
    _g("RTX 3060", ["RTX 3060"], (8, 6), "ampere", 28, 12, 360, 170,
       fp32=12.7, tf32=25.4, bf16=25.4, fp16=25.4),
    _g("RTX 3050", ["RTX 3050"], (8, 6), "ampere", 20, 8, 224, 130,
       fp32=9.1, tf32=18.2, bf16=18.2, fp16=18.2),
    # ------------------------------------------------------ Ada (8.9)
    _g("RTX 6000 Ada", ["RTX 6000 Ada"], (8, 9), "ada", 96, 48, 960, 300,
       fp32=72.9, tf32=145.7, bf16=145.7, fp16=145.7, fp8=291.4),
    _g("RTX 5000 Ada", ["RTX 5000 Ada"], (8, 9), "ada", 64, 32, 576, 250,
       fp32=49.2, tf32=97.9, bf16=97.9, fp16=97.9, fp8=195.7),
    _g("RTX 4000 Ada", ["RTX 4000 Ada"], (8, 9), "ada", 60, 20, 360, 150,
       fp32=26.7, tf32=53.3, bf16=53.3, fp16=53.3, fp8=106.6),
    _g("L40S", ["L40S"], (8, 9), "ada", 84, 48, 864, 350,
       fp32=91.6, tf32=183, bf16=183, fp16=183, fp8=366,
       note="fp8-ready datacenter Ada"),
    _g("L40", ["L40"], (8, 9), "ada", 84, 48, 864, 300,
       fp32=90.5, tf32=181, bf16=181, fp16=181, fp8=362),
    _g("L20", ["L20"], (8, 9), "ada", 84, 48, 864, 275,
       fp32=59.8, tf32=119.5, bf16=119.5, fp16=119.5, fp8=239,
       note="China-market Ada"),
    _g("L4", ["L4"], (8, 9), "ada", 58, 24, 300, 72,
       fp32=30.3, tf32=60.5, bf16=121, fp16=121, fp8=242),
    _g("RTX 4090", ["RTX 4090"], (8, 9), "ada", 128, 24, 1008, 450,
       fp32=82.6, tf32=82.6, bf16=165.2, fp16=165.2, fp8=330.3,
       note="no NVLink P2P on multi-GPU rigs"),
    _g("RTX 4090 D", ["4090 D"], (8, 9), "ada", 114, 24, 1008, 425,
       fp32=73.5, tf32=73.5, bf16=147, fp16=147, fp8=294),
    _g("RTX 4080", ["RTX 4080"], (8, 9), "ada", 76, 16, 717, 320,
       fp32=48.7, tf32=48.7, bf16=97.4, fp16=97.4, fp8=194.7),
    _g("RTX 4070 Ti", ["4070 Ti"], (8, 9), "ada", 60, 12, 504, 285,
       fp32=44.1, tf32=44.1, bf16=88.2, fp16=88.2, fp8=176.4),
    _g("RTX 4070", ["RTX 4070"], (8, 9), "ada", 46, 12, 504, 200,
       fp32=29.1, tf32=29.1, bf16=58.2, fp16=58.2, fp8=116.5),
    _g("RTX 4060 Ti", ["4060 Ti"], (8, 9), "ada", 36, 16, 288, 160,
       fp32=22.6, tf32=22.6, bf16=45.2, fp16=45.2, fp8=90.5),
    _g("RTX 4060", ["RTX 4060"], (8, 9), "ada", 24, 8, 272, 115,
       fp32=15.1, tf32=15.1, bf16=30.2, fp16=30.2, fp8=60.4),
    # ----------------------------------------------------- Hopper (9.0)
    _g("H200", ["H200"], (9, 0), "hopper", 132, 141, 4800, 700,
       fp32=67, tf32=495, bf16=990, fp16=990, fp8=1979),
    _g("H100 SXM", ["H100-SXM", "H100 80GB HBM3"], (9, 0), "hopper", 132,
       80, 3350, 700, fp32=67, tf32=495, bf16=990, fp16=990, fp8=1979,
       note="reference for fp8 + FA3 kernels"),
    _g("H100 PCIe", ["H100 PCIe", "H100 NVL"], (9, 0), "hopper", 114, 80,
       2039, 350, fp32=51, tf32=378, bf16=756, fp16=756, fp8=1513),
    _g("H800", ["H800"], (9, 0), "hopper", 132, 80, 2039, 700,
       fp32=67, tf32=495, bf16=990, fp16=990, fp8=1979,
       note="H100 compute, NVLink capped"),
    _g("GH200", ["GH200"], (9, 0), "hopper", 132, 96, 4000, 600,
       fp32=67, tf32=495, bf16=990, fp16=990, fp8=1979,
       note="Grace-Hopper: 480 GB unified LPDDR5X on some SKUs"),
    _g("H20", ["H20"], (9, 0), "hopper", 78, 96, 4000, 480,
       fp32=22, tf32=74, bf16=148, fp16=148, fp8=296,
       note="compute-lean, bandwidth-rich; fp8 still worthwhile"),
    # ------------------------------------------- Blackwell datacenter (10.x)
    _g("B300", ["B300", "GB300"], (10, 3), "bw_dc", 160, 288, 8000, 1400,
       fp32=80, tf32=1000, bf16=2000, fp16=2000, fp8=4000, fp4=15000,
       note="Blackwell Ultra: FP4-dense doubled vs B200"),
    _g("B200", ["B200", "GB200"], (10, 0), "bw_dc", 148, 192, 8000, 1200,
       fp32=80, tf32=1100, bf16=2250, fp16=2250, fp8=4500, fp4=9000),
    _g("B100", ["B100"], (10, 0), "bw_dc", 144, 192, 8000, 1000,
       fp32=70, tf32=880, bf16=1800, fp16=1800, fp8=3600, fp4=7000),
    # -------------------------------------------- Blackwell consumer (12.x)
    _g("RTX PRO 6000", ["RTX PRO 6000"], (12, 0), "bw_consumer", 188, 96,
       1792, 600, fp32=125, tf32=125, bf16=250, fp16=250, fp8=500,
       fp4=2000, note="workstation Blackwell, 96 GB GDDR7 (est. peaks)"),
    _g("RTX 5090", ["RTX 5090"], (12, 0), "bw_consumer", 170, 32, 1792,
       575, fp32=104.8, tf32=104.8, bf16=209.5, fp16=209.5, fp8=419,
       fp4=1676, note="CUDA 12.8+ toolchain required (sm_120)"),
    _g("RTX 5080", ["RTX 5080"], (12, 0), "bw_consumer", 84, 16, 960, 360,
       fp32=56.3, tf32=56.3, bf16=112.6, fp16=112.6, fp8=225, fp4=900),
    _g("RTX 5070 Ti", ["5070 Ti"], (12, 0), "bw_consumer", 70, 16, 896,
       300, fp32=44.5, tf32=44.5, bf16=89, fp16=89, fp8=178, fp4=712),
    _g("RTX 5070", ["RTX 5070"], (12, 0), "bw_consumer", 48, 12, 672, 250,
       fp32=30.9, tf32=30.9, bf16=61.8, fp16=61.8, fp8=123.5, fp4=494),
    _g("RTX 5060 Ti", ["5060 Ti"], (12, 0), "bw_consumer", 36, 16, 448,
       180, fp32=24.6, tf32=24.6, bf16=49.2, fp16=49.2, fp8=98.5, fp4=394),
    _g("RTX 5060", ["RTX 5060"], (12, 0), "bw_consumer", 30, 8, 320, 145,
       fp32=18.9, tf32=18.9, bf16=37.8, fp16=37.8, fp8=75.5, fp4=302),
    _g("RTX 5050", ["RTX 5050"], (12, 0), "bw_consumer", 20, 8, 224, 130,
       fp32=15.4, tf32=15.4, bf16=30.7, fp16=30.7, fp8=61.5, fp4=246,
       note="entry Blackwell; CUDA 12.8+ required here too"),
]

# Longest names first so "RTX 5070 Ti" wins over "RTX 5070", "A100 80GB"
# over "A100", "H100 SXM" over "H100", etc.
_GPU_DB_SORTED = sorted(GPU_DB, key=lambda e: -len(e["name"]))


# ===========================================================================
# Per-GPU tuning database  (THE custom settings, researched per card)
# ===========================================================================
# Every card in GPU_DB gets an explicit entry here -- no family-generic
# hand-me-downs.  Fields (missing keys fall back to _FAMILY_TUNE_DEFAULTS):
#
#   kernel_eff     fraction of the datasheet dense peak a well-tuned LARGE
#                  GEMM actually sustains on this die (measured-class values
#                  from cuBLASLt/CUTLASS behaviour per architecture).
#   precision_pref ordered precision ladder; first die-supported entry wins.
#   compile_mode   torch.compile mode that pays off on this silicon:
#                  "reduce-overhead" (CUDA graphs -- small-model launch-bound
#                  wins), "max-autotune" (deep GEMM pipelining on big dies),
#                  "default", or None (don't compile).  NOTE 2026-08: CUDA-
#                  graph modes are downgraded at build time (see the compile
#                  guard in build_profile) - cudagraph_trees crashes on this
#                  model's MoE host-sync graph breaks.
#   sdpa_order     attention kernel priority for SDPA backend gating.
#   env            per-architecture allocator/NCCL/cuBLAS environment knobs.
#   data_feed      dataloader pacing that keeps this host class fed without
#                  OOM: workers / chunk_docs / pause_s suggestions.
#   ckpt           "auto" (memory model decides) | "on" | "off" bias.
#   pin_memory     pinned staging pays off? (off for unified-memory parts).
#   ra             architecture rationale, surfaced in reports/notes.
#
_FAMILY_TUNE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "precision_pref": None,          # None -> engine's die-driven ladder
    "compile_mode": None,            # None -> engine decides
    "sdpa_order": None,
    "env": {},
    "data_feed": None,               # None -> engine's VRAM/CPU-tier default
    "ckpt": "auto",
    "pin_memory": True,
    "kernel_eff": None,              # None -> family kernel efficiency below
}

_FAMILY_KERNEL_EFF = {                # conservative large-GEMM ceilings
    "pascal": 0.30, "volta": 0.50, "turing": 0.46, "ampere": 0.55,
    "ada": 0.58, "hopper": 0.66, "bw_dc": 0.62, "bw_consumer": 0.58,
}

_TUNE_DB: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------- Pascal (6.x)
    "Tesla P100": dict(
        kernel_eff=0.32, precision_pref=["fp16", "fp32"],
        compile_mode=None, data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="56 SM, 732 GB/s HBM2 but NO tensor cores; fp16 still runs at 2x "
           "on CUDA cores (42 TF) and halves bandwidth pressure, so the "
           "ladder tries fp16+GradScaler before fp32."),
    "Tesla P40": dict(
        kernel_eff=0.30, precision_pref=["fp32"], compile_mode=None,
        data_feed=dict(workers=4, chunk_docs=400, pause_s=0.5),
        ra="347 GB/s and no fp16 acceleration at all: fp32-only, MFU ceiling "
           "is single-digit percent on modern workloads."),
    "Tesla P4": dict(
        kernel_eff=0.28, precision_pref=["fp32"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="75 W inference card, 192 GB/s: keep batches small and the data "
           "feed gentle; fp32-only."),
    # -------------------------------------------------------- Volta (7.0)
    "Tesla V100": dict(
        kernel_eff=0.52, precision_pref=["fp16"], compile_mode="default",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="640 fp16 tensor cores at 125 TF with 900 GB/s HBM2: fp16+"
           "GradScaler is the only fast path; first-gen TCs prefer plain "
           "compile (max-autotune gains are marginal on sm_70)."),
    # ------------------------------------------------------- Turing (7.5)
    "Tesla T4": dict(
        kernel_eff=0.42, precision_pref=["fp16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="40 SM at a 70 W cap and only 320 GB/s: bandwidth-starved and "
           "launch-bound on small models. fp16 TCs mandatory; inductor "
           "kernel fusion (default mode) is the lever - CUDA graphs are "
           "OFF (cudagraph_trees crashes on the MoE graph breaks)."),
    "RTX 2080 Ti": dict(
        kernel_eff=0.50, precision_pref=["fp16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="68 SM, 616 GB/s: healthy Turing. fp16 TCs + CUDA graphs; the "
           "11 GB frame tolerates no-ckpt batches that a T4 cannot."),
    "RTX 2080": dict(
        kernel_eff=0.48, precision_pref=["fp16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="46 SM, 448 GB/s: same recipe as 2080 Ti with tighter VRAM."),
    "RTX 2070": dict(
        kernel_eff=0.46, precision_pref=["fp16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=400, pause_s=0.5),
        ra="36 SM, 448 GB/s budget Turing: fp16 TCs, CUDA graphs, small "
           "batches."),
    "GTX 1660 Ti": dict(
        kernel_eff=0.30, precision_pref=["fp32"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="Turing without tensor cores: fp16 runs at fp32 rate, so stay in "
           "fp32 and keep expectations at pre-Ampere levels."),
    "GTX 1650": dict(
        kernel_eff=0.26, precision_pref=["fp32"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=200, pause_s=1.0),
        ra="128 GB/s, 16 SM, no TCs: bring-up card only."),
    # ---------------------------------------------- Ampere datacenter (8.0)
    "A100 80GB": dict(
        kernel_eff=0.62, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="108 SM, 2.0 TB/s HBM2e, 312 TF bf16 with FA2: the reference "
           "trainer. max-autotune pipelines the 640-wide GEMMs; NVLink makes "
           "DDP allreduce nearly free."),
    "A100 40GB": dict(
        kernel_eff=0.60, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="Same die as A100-80 at 1.55 TB/s and half the frame: identical "
           "recipe, the memory model will pick smaller micro-batches."),
    "A800 80GB": dict(
        kernel_eff=0.60, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="A100 derivative with NVLink capped at 400 GB/s: training "
           "settings identical, multi-card scaling slightly lower."),
    "A30": dict(
        kernel_eff=0.55, precision_pref=["bf16"], compile_mode="default",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="56 SM, 933 GB/s: half an A100; bf16 + FA2, compile gains are "
           "modest at this width."),
    # ------------------------------------------------ Ampere rest (8.6/8.7)
    "A10G": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="80 SM at 150 W, 600 GB/s: bf16 + FA2; CUDA graphs help the "
           "small-model launch bound."),
    "A10": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="72 SM, 600 GB/s, 150 W datacenter Ampere: same recipe as A10G "
           "(bf16 + FA2), CUDA graphs shave the small-model launch tax."),
    "A40": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="default",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="107 SM, 696 GB/s workstation Ampere: bf16 + FA2, plenty of "
           "frame for no-ckpt mid-size batches."),
    "RTX A6000": dict(
        kernel_eff=0.54, precision_pref=["bf16"], compile_mode="default",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="84 SM, 768 GB/s, 48 GB: the no-compromise workstation card; "
           "48 GB frame lets the memory model skip checkpointing entirely."),
    "RTX A5000": dict(
        kernel_eff=0.50, precision_pref=["bf16"], compile_mode="default",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="64 SM, 768 GB/s, 24 GB: balanced; bf16 + FA2."),
    "RTX A4000": dict(
        kernel_eff=0.46, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="48 SM, 448 GB/s, single-slot: small batches + CUDA graphs."),
    "A2": dict(
        kernel_eff=0.35, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="60 W edge card, 200 GB/s: MFU target is unrealistic here; "
           "settings optimise for steady-state throughput instead."),
    "Jetson AGX Orin 64GB": dict(
        kernel_eff=0.30, precision_pref=["bf16"], compile_mode=None,
        pin_memory=False, data_feed=dict(workers=2, chunk_docs=200, pause_s=1.0),
        ra="Unified memory with the CPU: pinned staging is useless, keep "
           "workers at 2 and let the 205 GB/s set the pace."),
    # ------------------------------------------------- Ampere consumer
    "RTX 3090 Ti": dict(
        kernel_eff=0.55, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="84 SM, 1.0 TB/s: the best consumer Ampere; bf16 + FA2 + CUDA "
           "graphs."),
    "RTX 3090": dict(
        kernel_eff=0.54, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="82 SM, 936 GB/s, 24 GB: classic trainer; remember GeForce "
           "multi-card rigs get NCCL_P2P_DISABLE."),
    "RTX 3080 Ti": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="68 SM, 760 GB/s, 12 GB frame: same silicon class as 3090 with "
           "half the VRAM -- the memory model compensates."),
    "RTX 3080": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="68 SM, 760 GB/s, 10 GB: watch the frame; graphs keep the small "
           "GEMMs dense."),
    "RTX 3070 Ti": dict(
        kernel_eff=0.48, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="48 SM, 608 GB/s, 8 GB: small-die Ampere; CUDA graphs + tight "
           "batches."),
    "RTX 3070": dict(
        kernel_eff=0.47, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="46 SM, 448 GB/s, 8 GB: same story, slightly less bandwidth."),
    "RTX 3060": dict(
        kernel_eff=0.42, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="28 SM, 360 GB/s but 12 GB of VRAM: batches stay small, the "
           "frame at least lets eval run without cache churn."),
    "RTX 3050": dict(
        kernel_eff=0.36, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=200, pause_s=1.0),
        ra="20 SM, 224 GB/s entry card: settings optimise for not starving "
           "the GPU; MFU target likely out of reach."),
    # ------------------------------------------------------ Ada (8.9)
    "RTX 6000 Ada": dict(
        kernel_eff=0.58, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1000, pause_s=0.3),
        ra="96 SM, 960 GB/s, 48 GB: full-fat Ada; fp8 exists in silicon but "
           "the training stack is younger than on Hopper -- bf16 first."),
    "RTX 5000 Ada": dict(
        kernel_eff=0.54, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="64 SM, 576 GB/s, 32 GB: workstation Ada, same recipe."),
    "RTX 4000 Ada": dict(
        kernel_eff=0.50, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="60 SM, 360 GB/s single-slot: CUDA graphs matter more than "
           "autotune at this bandwidth."),
    "L40S": dict(
        kernel_eff=0.58, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1000, pause_s=0.3),
        ra="84 SM, 864 GB/s, fp8 TCs: bf16 is the reliable default; install "
           "TE/torchao and --precision fp8 roughly doubles matmul rate."),
    "L40": dict(
        kernel_eff=0.57, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1000, pause_s=0.3),
        ra="Same die as L40S without the fp8 firmware wink: bf16 recipe."),
    "L20": dict(
        kernel_eff=0.54, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="China-market Ada at 275 W: same die, slightly lower clocks."),
    "L4": dict(
        kernel_eff=0.45, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="58 SM at a 72 W cap, only 300 GB/s: bandwidth is the wall; "
           "graphs + modest batches extract what's there."),
    "RTX 4090": dict(
        kernel_eff=0.60, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1000, pause_s=0.3),
        ra="128 SM, 1.0 TB/s, 165 TF bf16: the fastest consumer trainer; "
           "max-autotune pays off at this width, fp8 opt-in via torchao."),
    "RTX 4090 D": dict(
        kernel_eff=0.57, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1000, pause_s=0.3),
        ra="114 SM variant of the 4090: identical recipe, ~11% less peak."),
    "RTX 4080": dict(
        kernel_eff=0.54, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="76 SM, 717 GB/s, 16 GB: strong die, tight frame -- graphs and "
           "the memory model do the fine-tuning."),
    "RTX 4070 Ti": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="60 SM, 504 GB/s, 12 GB: mid Ada; CUDA graphs + small batches."),
    "RTX 4070": dict(
        kernel_eff=0.48, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="46 SM, 504 GB/s, 12 GB: same class a notch down."),
    "RTX 4060 Ti": dict(
        kernel_eff=0.42, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="36 SM, 288 GB/s: bandwidth-bound; compile's graph buffers "
           "outweigh gains at 16 GB only on paper -- kept off."),
    "RTX 4060": dict(
        kernel_eff=0.38, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=200, pause_s=1.0),
        ra="24 SM, 272 GB/s entry Ada: gentle pacing, small batches."),
    # ----------------------------------------------------- Hopper (9.0)
    "H200": dict(
        kernel_eff=0.68, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "32768",
             "TORCH_CUDNN_V8_API_ENABLED": "1"},
        data_feed=dict(workers=8, chunk_docs=1500, pause_s=0.2),
        ra="132 SM at 4.8 TB/s HBM3e: the bandwidth king. fp8+TE first, FA2/"
           "FA3, big cuBLASLt workspaces for Hopper GEMM pipelines."),
    "H100 SXM": dict(
        kernel_eff=0.66, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "32768",
             "TORCH_CUDNN_V8_API_ENABLED": "1"},
        data_feed=dict(workers=8, chunk_docs=1500, pause_s=0.2),
        ra="The 990 TF reference: fp8 with TE/torchao doubles matmul rate "
           "again; FA3-class attention; NVLink keeps DDP quiet."),
    "H100 PCIe": dict(
        kernel_eff=0.62, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "32768"},
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="Hopper compute at 2.0 TB/s and 350 W: same fp8-first ladder, "
           "thermal headroom is the practical limit."),
    "H800": dict(
        kernel_eff=0.64, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "32768"},
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="H100 compute with capped NVLink: local settings identical; "
           "multi-card scale-out suffers, single-card does not."),
    "GH200": dict(
        kernel_eff=0.64, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune", pin_memory=False,
        data_feed=dict(workers=4, chunk_docs=800, pause_s=0.3),
        ra="Grace-Hopper: NVLink-C2C to LPDDR5X -- pinned host staging is "
           "pointless, keep the feed local and let 4 TB/s HBM breathe."),
    "H20": dict(
        kernel_eff=0.50, precision_pref=["fp8", "bf16"],
        compile_mode="default",
        data_feed=dict(workers=6, chunk_docs=1000, pause_s=0.3),
        ra="78 SM, compute-lean (148 TF bf16) but 4 TB/s: the rare card "
           "where BIG batches are free -- the MFU seeker will push the "
           "micro-batch until compute saturates."),
    # ------------------------------------------- Blackwell datacenter (10.x)
    "B300": dict(
        kernel_eff=0.62, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "65536",
             "TORCH_CUDNN_V8_API_ENABLED": "1"},
        data_feed=dict(workers=8, chunk_docs=2000, pause_s=0.2),
        ra="Blackwell Ultra, 8 TB/s and 15 PF fp4-dense: fp8+TE first; cuDNN "
           "SDPA attention; young toolchain keeps kernel_eff honest at 0.62."),
    "B200": dict(
        kernel_eff=0.62, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "65536",
             "TORCH_CUDNN_V8_API_ENABLED": "1"},
        data_feed=dict(workers=8, chunk_docs=2000, pause_s=0.2),
        ra="2.25 PF bf16, 8 TB/s: same recipe as B300; CUDA 12.8+ toolchain "
           "and driver >= 570 are hard gates."),
    "B100": dict(
        kernel_eff=0.60, precision_pref=["fp8", "bf16"],
        compile_mode="max-autotune",
        env={"CUBLASLT_WORKSPACE_SIZE": "65536"},
        data_feed=dict(workers=8, chunk_docs=2000, pause_s=0.2),
        ra="The 1.8 PF sibling: identical stack, slightly leaner peaks."),
    # -------------------------------------------- Blackwell consumer (12.x)
    "RTX PRO 6000": dict(
        kernel_eff=0.60, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1500, pause_s=0.2),
        ra="188 SM, 96 GB GDDR7: workstation Blackwell -- bf16 is the "
           "mature path, the 96 GB frame removes checkpointing entirely "
           "for mid-size variants."),
    "RTX 5090": dict(
        kernel_eff=0.58, precision_pref=["bf16"], compile_mode="max-autotune",
        data_feed=dict(workers=8, chunk_docs=1200, pause_s=0.2),
        ra="170 SM, 1.79 TB/s GDDR7: fastest consumer card; bf16 + cuDNN "
           "SDPA; CUDA 12.8+ wheels are mandatory on sm_120."),
    "RTX 5080": dict(
        kernel_eff=0.52, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=6, chunk_docs=800, pause_s=0.3),
        ra="84 SM, 960 GB/s, 16 GB: consumer Blackwell; graphs beat "
           "autotune on this frame size."),
    "RTX 5070 Ti": dict(
        kernel_eff=0.50, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=600, pause_s=0.5),
        ra="70 SM, 896 GB/s, 16 GB: same recipe one notch down."),
    "RTX 5070": dict(
        kernel_eff=0.45, precision_pref=["bf16"], compile_mode="reduce-overhead",
        data_feed=dict(workers=4, chunk_docs=500, pause_s=0.5),
        ra="48 SM, 672 GB/s, 12 GB: mid consumer Blackwell."),
    "RTX 5060 Ti": dict(
        kernel_eff=0.42, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=300, pause_s=1.0),
        ra="36 SM, 448 GB/s: bandwidth-bound entry card; gentle pacing."),
    "RTX 5060": dict(
        kernel_eff=0.38, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=250, pause_s=1.0),
        ra="30 SM, 320 GB/s, 8 GB: small everything; settings prioritise "
           "staying resident over raw MFU."),
    "RTX 5050": dict(
        kernel_eff=0.34, precision_pref=["bf16"], compile_mode=None,
        data_feed=dict(workers=2, chunk_docs=200, pause_s=1.0),
        ra="20 SM entry Blackwell at 224 GB/s: MFU target is aspirational "
           "here; the plan reports the honest ceiling."),
}

# Backfill: any DB card missing above still gets explicit defaults so the
# self-test's per-card completeness check is meaningful.
for _e in GPU_DB:
    _TUNE_DB.setdefault(_e["name"], {})


def _tune_for(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the card's explicit tuning over the family defaults."""
    fam_eff = _FAMILY_KERNEL_EFF[entry["family"]]
    merged = dict(_FAMILY_TUNE_DEFAULTS)
    merged["kernel_eff"] = fam_eff
    merged.update(_TUNE_DB.get(entry["name"], {}))
    return merged


def list_known_gpus() -> List[str]:
    return [e["name"] for e in GPU_DB]


# ===========================================================================
# Detection
# ===========================================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("-", " ").replace("_", " "))


def match_gpu_db(name: str) -> Optional[Dict[str, Any]]:
    """Fuzzy-match a device string against the DB (longest needles first)."""
    if not name:
        return None
    hay = _norm(name)
    for entry in _GPU_DB_SORTED:
        for needle in [entry["name"], *entry["alias"]]:
            if re.search(r"\b" + re.escape(_norm(needle)) + r"\b", hay):
                return entry
    return None


@dataclass
class Detection:
    name: str
    source: str                     # nvidia-smi | torch | simulate | none
    match: str                      # db | generic | none
    cc: Optional[Tuple[int, int]] = None
    sms: Optional[int] = None
    vram_gb: Optional[float] = None
    driver: Optional[str] = None
    gpu_count: int = 1


def probe_nvidia_smi() -> Optional[Detection]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    name_part, cc_part, mem_part, drv_part = (lines[0].split(",") + ["", "",
                                                              "", ""])[:4]
    det = Detection(name=name_part.strip(), source="nvidia-smi", match="none",
                    gpu_count=len(lines), driver=drv_part.strip() or None)
    try:
        maj, _, mnr = cc_part.strip().partition(".")
        det.cc = (int(maj), int(mnr or 0))
    except ValueError:
        det.cc = None
    try:
        det.vram_gb = round(float(mem_part) / 1024.0, 1)
    except ValueError:
        det.vram_gb = None
    entry = match_gpu_db(det.name)
    if entry is not None:
        det.match = "db"
        det.sms = det.sms or entry["sms"]
        det.cc = det.cc or tuple(entry["cc"])
        det.vram_gb = det.vram_gb or float(entry["vram"])
    return det


def probe_torch(index: int = 0) -> Optional[Detection]:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(index)
        det = Detection(
            name=getattr(props, "name", "") or "unknown",
            source="torch", match="none",
            cc=(props.major, props.minor),
            sms=getattr(props, "multi_processor_count", None),
            vram_gb=round(props.total_memory / (1 << 30), 1),
            gpu_count=torch.cuda.device_count(),
        )
        entry = match_gpu_db(det.name)
        if entry is not None:
            det.match = "db"
        return det
    except Exception:                                    # pragma: no cover
        return None


def probe_host() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cpu_count": os.cpu_count() or 4,
                            "host_ram_gb": None}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["host_ram_gb"] = round(
                        int(line.split()[1]) / (1 << 20), 1)
                    break
    except (OSError, ValueError):
        pass
    return info


def probe_extras() -> Dict[str, Any]:
    """Package/toolchain probes that do NOT import heavy modules."""
    from importlib.util import find_spec

    def has(mod: str) -> bool:
        try:
            return find_spec(mod) is not None
        except (ImportError, ValueError):
            return False

    info: Dict[str, Any] = {
        "transformer_engine": has("transformer_engine"),
        "torchao": has("torchao"),
        "flash_attn": has("flash_attn"),
        "flash_attn_3": has("flash_attn_interface"),
        "torch": None, "torch_cuda": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _family_for_cc(cc: Tuple[int, int]) -> str:
    maj, mnr = cc
    if maj == 6:
        return "pascal"
    if maj == 7:
        return "volta" if mnr == 0 else "turing"
    if maj == 8:
        return "ada" if mnr == 9 else "ampere"
    if maj == 9:
        return "hopper"
    if maj == 10:
        return "bw_dc"
    if maj >= 12:
        return "bw_consumer"
    return "ampere"                                      # safest modern guess


def synthesize_entry(cc: Tuple[int, int], sms: int,
                     vram_gb: float) -> Dict[str, Any]:
    """Best-effort spec for cards missing from the DB (peaks marked est)."""
    family = _family_for_cc(cc)
    per_sm = _PEAKS_PER_SM.get(cc) or _PEAKS_PER_SM[(8, 6)]
    scale = float(sms or 80)
    peaks = dict(
        fp32=round(0.15 * scale, 1), tf32=round(per_sm[0] * scale, 1),
        bf16=round(per_sm[1] * scale, 1), fp16=round(per_sm[2] * scale, 1),
        fp8=round(per_sm[3] * scale, 1), fp4=round(per_sm[4] * scale, 1),
    )
    return _g(f"Unknown cc{cc[0]}.{cc[1]}", [], cc, family, sms or 80,
              vram_gb or 16, 500, 250, fp32=peaks["fp32"],
              tf32=peaks["tf32"], bf16=peaks["bf16"], fp16=peaks["fp16"],
              fp8=peaks["fp8"], fp4=peaks["fp4"],
              note="synthesised from compute capability (est. peaks)")


# ===========================================================================
# Tuned-profile builder
# ===========================================================================
_PER_SEQ_GB = {"100m": 1.2, "300m": 1.9, "500m": 2.8, "1b": 4.5,
               "3b": 8.5, "7b": 16.0, "10b": 22.0}
_NOMINAL_PARAMS_B = {"100m": 0.10, "300m": 0.30, "500m": 0.52, "1b": 1.05,
                     "3b": 3.1, "7b": 7.2, "10b": 10.5}
_TARGET_SEQS = {"100m": 64, "300m": 48, "500m": 32, "1b": 32,
                "3b": 64, "7b": 64, "10b": 64}
_DEF_SEQ = {"100m": 2048, "300m": 2048}

# ---------------------------------------------------------------------------
# MFU roofline engine -- aims every card at an explicit MFU target (30%).
#
# The model (documented, deliberately conservative, torch-free):
#
#   step_flops(B)   = 3 x fwd_flops_per_token x B_tok x (1.33 if ckpt)
#                     fwd_flops_per_token ~= _FWD_TOK[v] x (1 + seq/8192)
#                     (3x = forward + backward; 1.33 = checkpoint recompute)
#   traffic(B)      = weights_fwd_read + grad_write   (bytes/param x P x 2.2)
#                   + AdamW states / accum            (16 bytes/param)
#                   + DDP allreduce / accum           (world > 1)
#                   + activation write+read           (_PER_SEQ_GB based)
#   AI(B)           = step_flops / traffic            (FLOP per byte moved)
#   ridge           = peak_TFLOPS x 1000 / bw_GBps    (FLOP per byte)
#   wave(B)         = clamp(B_tok / (SMs x 512), 0.45, 1.0)
#                     (a d_model-wide GEMM needs ~512 token-rows per SM to
#                     fill the die -- small models on big dies derate hard)
#   MFU(B)          = min(1, AI/ridge) x kernel_eff x e_attn x e_compile
#                     x e_scaler x e_ckpt x wave(B)
#
# The seeker sweeps micro-batch (and the ckpt on/off option) and picks the
# setting that maximises projected MFU, reporting the gap to the 30% target
# honestly -- if the silicon cannot reach it, the plan says so and why.
# ---------------------------------------------------------------------------
_MFU_TARGET_DEFAULT = 0.30
_FWD_TOK = {"100m": 0.30e9, "300m": 0.52e9, "500m": 0.73e9, "1b": 1.75e9,
            "3b": 4.0e9, "7b": 8.8e9, "10b": 11.5e9}
_COMPILE_GAIN = {"max-autotune": 1.10, "reduce-overhead": 1.18,
                 "max-autotune-no-cudagraphs": 1.09,
                 "default": 1.04, None: 1.0}
#: Fixed multiplier for framework overheads the per-GPU model does not
#: otherwise capture: MoE dispatch/combine kernels, recurrent-loop launch
#: churn, GradScaler events, DDP gradient-sync gaps, Python-side step
#: overhead. Calibrated so T4-class projections land near observed reality
#: instead of datasheet fantasy.
_WORKLOAD_OVERHEAD = 0.72


def _bytes_per_param(precision: str) -> float:
    return {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0}.get(precision, 2.0)


def _attn_eff(use_flash: bool, family: str) -> float:
    if use_flash:
        return 1.0
    if family in ("pascal",):
        return 0.70                     # math SDPA only
    if family in ("volta", "turing"):
        return 0.85                     # mem-efficient SDPA
    return 0.95                         # cuDNN SDPA fallback


def _project_mfu(batch: int, seq: int, vkey: str, entry: Dict[str, Any],
                 peak_tflops: float, precision: str, world: int, accum: int,
                 ckpt: bool, use_flash: bool, compile_mode: Optional[str],
                 kernel_eff: float) -> Dict[str, float]:
    """Roofline projection for one (batch, ckpt) configuration."""
    s = float(seq)
    b_tok = float(batch) * s
    p_total = _NOMINAL_PARAMS_B[vkey]
    recompute = 1.33 if ckpt else 1.0
    step_flops = 3.0 * _FWD_TOK[vkey] * (1.0 + s / 8192.0) * b_tok * recompute

    bpp = _bytes_per_param(precision)
    w_bytes = bpp * p_total * 1e9 * 2.2                 # read fwd + write grad
    opt_bytes = 16.0 * p_total * 1e9 / max(accum, 1)    # AdamW once per step
    ddp_bytes = (bpp * p_total * 1e9 / max(accum, 1)
                 if world > 1 else 0.0)
    act_scale = (_PER_SEQ_GB[vkey] * (s / 4096.0) ** 1.6
                 * (0.4 if ckpt else 1.0))
    act_bytes = act_scale * 1e9 * float(batch)
    traffic = max(w_bytes + opt_bytes + ddp_bytes + act_bytes, 1.0)

    ai = step_flops / traffic                           # FLOP / byte
    ridge = max(peak_tflops, 1e-6) * 1000.0 / max(entry["bw"], 1)
    wave = min(1.0, max(0.45, b_tok / max(entry["sms"] * 512, 1)))
    e_compile = _COMPILE_GAIN.get(compile_mode, 1.0)
    e_scaler = 0.98 if precision == "fp16" else 1.0
    e_ckpt = 0.90 if ckpt else 1.0
    adjust = (kernel_eff * _attn_eff(use_flash, entry["family"])
              * e_compile * e_scaler * e_ckpt * _WORKLOAD_OVERHEAD)
    mfu = min(1.0, ai / ridge) * adjust * wave
    return dict(ai=ai, ridge=ridge, wave=wave, mfu=min(max(mfu, 0.0), 1.0),
                compute_bound=bool(ai >= ridge))


def _mfu_seek(mem_cap_ckpt: int, mem_cap_nockpt: int, seq: int, vkey: str,
              entry: Dict[str, Any], peak: float, precision: str,
              world: int, accum: int, use_flash: bool,
              compile_mode: Optional[str], kernel_eff: float,
              ckpt_bias: str, target: float) -> Dict[str, Any]:
    """Sweep (ckpt option, micro-batch) and return the MFU-maximising plan.

    Option selection uses each option's full potential (best batch over its
    whole capacity sweep); within the winning option the returned batch is
    the SMALLEST one that already meets the target, keeping VRAM headroom
    -- or the potential-maximising batch when the target is unreachable.
    """
    options: List[Tuple[bool, int]] = []
    if ckpt_bias != "on" and mem_cap_nockpt >= 1:
        options.append((False, mem_cap_nockpt))
    if ckpt_bias != "off" and mem_cap_ckpt >= 1:
        options.append((True, mem_cap_ckpt))
    if not options:
        options = [(True, 1)]

    winning_best: Optional[Dict[str, Any]] = None
    winning_first_meet: Optional[Dict[str, Any]] = None
    winning_potential = -1.0
    for ckpt, cap in options:
        opt_best: Optional[Dict[str, Any]] = None
        opt_first_meet: Optional[Dict[str, Any]] = None
        for b in range(1, min(cap, 64) + 1):
            proj = _project_mfu(b, seq, vkey, entry, peak, precision, world,
                                accum, ckpt, use_flash, compile_mode,
                                kernel_eff)
            cand = dict(proj, batch=b, ckpt=ckpt)
            if (opt_best is None or proj["mfu"] > opt_best["mfu"] + 1e-9
                    or (abs(proj["mfu"] - opt_best["mfu"]) <= 1e-9
                        and b < opt_best["batch"])):
                opt_best = cand
            if (opt_first_meet is None and b >= 8
                    and proj["mfu"] >= target - 1e-9):
                opt_first_meet = cand          # keep sweeping: potential ends at cap
        if opt_best is None:
            continue
        if opt_best["mfu"] > winning_potential + 1e-9:
            winning_potential = opt_best["mfu"]
            winning_best = opt_best
            winning_first_meet = opt_first_meet

    best = winning_first_meet or winning_best
    assert best is not None
    best["target"] = target
    best["meets_target"] = bool(best["mfu"] >= target - 1e-9)
    return best


def detect_and_build_profile(variant: str = "500m",
                             seq_len: Optional[int] = None,
                             world_size: int = 1,
                             probe_pkgs: bool = True,
                             simulate: Optional[str] = None,
                             mfu_target: float = _MFU_TARGET_DEFAULT,
                             ) -> Dict[str, Any]:
    det = detect_gpu(simulate=simulate)
    host = probe_host()
    extras = probe_extras() if probe_pkgs else {}
    return build_profile(det, variant=variant, seq_len=seq_len,
                         world_size=world_size, host=host, extras=extras,
                         mfu_target=mfu_target)


def detect_gpu(simulate: Optional[str] = None) -> Detection:
    if simulate:
        det = Detection(name=simulate, source="simulate", match="none")
        entry = match_gpu_db(simulate)
        if entry is not None:
            det.match = "db"
            det.cc = tuple(entry["cc"])
            det.sms = entry["sms"]
            det.vram_gb = float(entry["vram"])
            det.driver = None
            det.gpu_count = 1
            return det
        # generic simulator: "cc=9.0,vram=80,sms=132"
        kv = dict(p.split("=", 1) for p in simulate.replace(";", ",").split(",")
                  if "=" in p)
        cc = tuple(int(x) for x in kv.get("cc", "8.0").split("."))
        det.cc = (cc[0], cc[1] if len(cc) > 1 else 0)
        det.sms = int(kv.get("sms", "80"))
        det.vram_gb = float(kv.get("vram", "40"))
        det.match = "generic"
        return det
    det = probe_nvidia_smi()
    if det is not None:
        return det
    det = probe_torch()
    if det is not None:
        return det
    return Detection(name="no GPU detected", source="none", match="none")


def build_profile(det: Detection, variant: str = "500m",
                  seq_len: Optional[int] = None, world_size: int = 1,
                  host: Optional[Dict[str, Any]] = None,
                  extras: Optional[Dict[str, Any]] = None,
                  mfu_target: float = _MFU_TARGET_DEFAULT,
                  ) -> Dict[str, Any]:
    host = host or {}
    extras = extras or {}
    vkey = variant.strip().lower()
    if vkey not in _PER_SEQ_GB:
        vkey = "500m"
    if seq_len is None or seq_len <= 0:
        seq_len = _DEF_SEQ.get(vkey, 4096)

    warnings: List[str] = []
    notes: List[str] = []
    env: Dict[str, str] = {"PYTORCH_CUDA_ALLOC_CONF":
                           "expandable_segments:True"}

    # ------------------------------------------------------- resolve silicon
    if det.match == "db":
        entry = match_gpu_db(det.name)
        assert entry is not None
        matched = entry["name"]
    elif det.cc is not None:
        entry = synthesize_entry(det.cc, det.sms or 80,
                                 det.vram_gb or 16.0)
        matched = None
    else:
        entry = synthesize_entry((8, 0), 108, 40.0)
        warnings.append("No NVIDIA GPU detected - profile assumes a generic "
                        "Ampere-class card; settings are informational only.")
        matched = None

    family = entry["family"]
    fam = FAMILIES[family]
    cc = tuple(entry["cc"])
    vram = float(det.vram_gb or entry["vram"])
    tune = _tune_for(entry)
    env.update(tune.get("env") or {})          # per-card cuBLAS/NCCL knobs
    mfu_target = min(max(float(mfu_target), 0.02), 0.95)

    # ----------------------------------------------------------- precision
    # Per-card ladder first (TUNE_DB precision_pref), die-support checked;
    # fallback is the family-driven rule for synthesised/unknown cards.
    def _supported(p: str) -> bool:
        if p == "fp32":
            return True
        if p == "fp16":
            return bool(entry.get("tc", fam["tc_fp16"]))
        if p == "bf16":
            return bool(fam["bf16"])
        if p == "fp8":
            return bool(fam["fp8"]) and cc[0] in (9, 10) and (
                extras.get("transformer_engine") or extras.get("torchao"))
        return False

    precision = None
    for p in (tune.get("precision_pref") or []):
        if _supported(p):
            precision = p
            break
    if precision is None:                      # unknown-card fallback ladder
        has_tc = bool(entry.get("tc", fam["tc_fp16"]))
        precision = "fp32" if not has_tc else (
            "bf16" if fam["bf16"] else "fp16")
        if not has_tc:
            notes.append("No tensor cores on this die: MFU ceiling is raw "
                         "fp32. Expect single-digit utilisation.")
        elif not fam["bf16"]:
            notes.append("Pre-Ampere tensor cores are fp16-only: the trainer "
                         "auto-enables a GradScaler for stable fp16 training.")
    if precision == "fp16":
        notes.append("fp16 + dynamic GradScaler selected (per-card ladder).")
    if precision == "fp8":
        notes.append("FP8 backend present (TE/torchao) and Hopper/Blackwell "
                     "silicon: auto-selected for ~2x matmul throughput. Pass "
                     "--precision bf16 to opt out.")
    if (precision == "bf16" and fam["fp8"] and cc[0] in (9, 10)):
        notes.append("fp8 supported in silicon; install transformer_engine "
                     "or torchao, then pass --precision fp8 to double the "
                     "matmul rate.")
    if cc[0] == 12 and fam["fp8"]:
        notes.append("Consumer Blackwell: fp8/nvfp4 stacks are still young - "
                     "bf16 is the reliable path today.")

    # ------------------------------------------------------ flash attention
    use_flash = bool(fam["flash"]) and (cc[0] >= 8)
    if cc[0] >= 10:
        use_flash = False
        notes.append("Blackwell: integrated FlashAttention wheels lag sm_100/"
                     "sm_120 - trainer falls back to SDPA (cuDNN attention "
                     "kernels). Flip --use_flash_attn on only if your "
                     "flash-attn wheel lists sm_1xx support.")
    elif not fam["flash"]:
        notes.append("SDPA memory-efficient kernels used (FlashAttention "
                     "requires Ampere / sm_80+).")

    # -------------------------------------------------- memory-derived sizes
    # Two capacity variants: with and without per-loop-step checkpointing.
    # The MFU seeker prefers the no-ckpt option whenever it both fits and
    # projects higher utilisation (recompute burns ~33% of the FLOPs).
    #
    # Static overhead (calibrated against a real 2x T4 OOM at 98% capacity,
    # 2026-08): fp16 weights+grads+fp32 AdamW moments scale with params
    # (16 bytes/param); the constant covers CUDA context, cuDNN workspaces,
    # GradScaler scratch and DDP buckets.  Multi-GPU adds NCCL staging
    # buffers (SHM transport when P2P is unavailable - cloud VMs);
    # CUDA-graph pools (compile_mode=reduce-overhead) reserve private
    # segments the allocator can never reuse for anything else.
    graphs_likely = tune.get("compile_mode") == "reduce-overhead"
    static_gb = (_NOMINAL_PARAMS_B[vkey] * 16.0 + 1.1
                 + (0.5 if int(world_size) > 1 else 0.0)
                 + (0.4 if graphs_likely else 0.0))
    per_seq_gb = _PER_SEQ_GB[vkey] * (seq_len / 4096.0) ** 1.6
    usable = vram * 0.88 - static_gb
    ckpt_bias = tune.get("ckpt", "auto")
    fits_at_all = usable > 0.6
    if not fits_at_all:
        warnings.append(
            f"{vkey}: weights+grads+AdamW states need ~{static_gb:.0f} GB but "
            f"the card has {vram:.0f} GB - single-GPU training cannot fit. "
            "Use torchrun --nproc_per_node>=2 with --dist_strategy fsdp, or "
            "drop to a smaller variant.")
        cap_ckpt, cap_nockpt = 1, 0
    else:
        cap_ckpt = max(1, min(64, int(usable / max(per_seq_gb / 2.5, 0.4))))
        # Headroom guard: a no-ckpt plan that "fits exactly" has zero margin
        # for allocator fragmentation, cuDNN algo workspace variance and
        # graph-pool growth - measured OOM happened at 98% of capacity.
        # Only 85% of the theoretical no-ckpt activation budget is offered
        # to the seeker; the ckpt option keeps the full budget because its
        # activation footprint is ~2.5x smaller by construction.
        cap_nockpt = max(1, min(64, int(usable * 0.85 / max(per_seq_gb, 0.4))))
    target = _TARGET_SEQS[vkey]

    # ------------------------------------------------------- data feed
    # Starved GPUs have zero MFU: the per-card feed suggestion keeps the
    # token buffer deep enough between pacing naps for this host class.
    ncpu = int(host.get("cpu_count") or 4)
    feed = dict(tune.get("data_feed")
                or dict(workers=4, chunk_docs=600, pause_s=0.5))
    num_workers = max(1, min(int(feed["workers"]), max(2, ncpu // 2)))
    feed_chunk = max(30, int(feed["chunk_docs"]))
    feed_pause = max(0.0, float(feed["pause_s"]))

    host_ram = host.get("host_ram_gb")
    if host_ram is not None and host_ram < 24:
        notes.append(f"Host RAM {host_ram:.0f} GB: consider --low_ram to cap "
                     "the streaming shuffle buffer.")

    # ------------------------------------------------------- multi-GPU env
    consumer = family in ("ampere", "ada", "bw_consumer") and cc[0] >= 8 \
        and not entry["name"].startswith(("A1", "A3", "A4", "GH", "H1", "H2",
                                          "B1", "B2", "B3"))
    if det.gpu_count and det.gpu_count > 1 and consumer:
        env["NCCL_P2P_DISABLE"] = "1"
        notes.append("Multi-GPU consumer rig: NCCL_P2P_DISABLE=1 set "
                     "(GeForce cards lack reliable P2P; avoids NCCL hangs).")
    ncpu_env = ncpu if ncpu >= 32 else None
    if ncpu_env:
        env["OMP_NUM_THREADS"] = "8"

    # --------------------------------------------------- toolchain gating
    cuda_min = _CUDA_MIN_BY_CC.get(cc)
    if cuda_min and cc[0] >= 10 or (cuda_min and cc[0] == 12):
        need = ".".join(str(x) for x in (12, 8))
        if extras.get("torch_cuda"):
            have = extras["torch_cuda"]
            have_v = tuple(int(x) for x in have.split(".")[:2])
            if have_v < (12, 8):
                warnings.append(
                    f"This is sm_{cc[0]}{cc[1]} silicon but the installed "
                    f"PyTorch ships CUDA {have}: kernels for Blackwell need "
                    f"CUDA {need}+ (torch cu128 wheels). Upgrade torch or "
                    "expect 'no kernel image' errors.")
        else:
            notes.append(f"Blackwell requires CUDA {need}+ toolchain "
                         "(torch cu128 builds and driver >= 570).")
    drv = det.driver
    if drv and cc[0] >= 10:
        try:
            if int(drv.split(".")[0]) < 570:
                warnings.append(f"Driver {drv} is older than 570; datacenter "
                                "Blackwell userspace needs >= 570.")
        except ValueError:
            pass

    # ------------------------------------------------------------ compile
    # Per-card mode: max-autotune on wide datacenter GEMMs, default elsewhere,
    # off where buffers cost more than kernels.
    compile_mode = tune.get("compile_mode")
    if compile_mode is None and fam["compile"]:
        compile_mode = ("max-autotune"
                        if family in ("hopper", "bw_dc")
                        or cc == (8, 0) else "default")
    if compile_mode is not None and vram < 8:
        compile_mode = None
        notes.append("Small VRAM: torch.compile's extra graph buffers "
                     "outweigh its speedup - left off.")
    if compile_mode == "reduce-overhead":
        # CUDA-graph trees guard (real 2x T4 crash, 2026-08): the MoE dispatch
        # loop host-syncs (boundaries.tolist(), moe.py) inside the compiled
        # region, dynamo splits the graph, and cudagraph_trees' checkpoint-
        # pool restore across segments raises "Expected curr_block->next ==
        # nullptr" from the caching allocator at step 1.  "default" keeps
        # the inductor kernel fusion without CUDA graphs.
        compile_mode = "default"
        notes.append("compile reduce-overhead downgraded to default: CUDA "
                     "graph trees crash on this model's MoE graph breaks "
                     "(torch allocator bug).")
    elif compile_mode == "max-autotune":
        # max-autotune enables cudagraphs via the same cudagraph_trees path
        # - keep the GEMM autotuning, drop the graphs.
        compile_mode = "max-autotune-no-cudagraphs"
        notes.append("compile max-autotune downgraded to max-autotune-no-"
                     "cudagraphs (same CUDA graph trees hazard).")

    peaks = entry["peaks"]
    mfu_key = {"fp32": "tf32" if fam["tf32"] else "fp32",
               "bf16": "bf16", "fp16": "fp16",
               "fp8": "fp8", "fp4": "fp4"}[precision]
    peak = float(peaks.get(mfu_key) or peaks.get("bf16")
                 or peaks.get("fp32") or 0.0)

    # ------------------------------------------------- MFU seek (2 passes)
    # accum depends on batch (global tokens/step target) and the roofline's
    # optimizer/DDP amortisation depends on accum: iterate twice to settle.
    accum = 8
    plan: Dict[str, Any] = {}
    for _pass in range(2):
        plan = _mfu_seek(
            mem_cap_ckpt=cap_ckpt, mem_cap_nockpt=cap_nockpt,
            seq=seq_len, vkey=vkey, entry=entry, peak=peak,
            precision=precision, world=max(int(world_size), 1), accum=accum,
            use_flash=use_flash, compile_mode=compile_mode,
            kernel_eff=float(tune["kernel_eff"]), ckpt_bias=ckpt_bias,
            target=mfu_target,
        )
        batch = plan["batch"]
        accum = max(1, min(256, round(target / max(batch, 1))))
    grad_checkpoint = bool(plan["ckpt"])

    settings = {
        "precision": precision,
        "use_flash_attn": use_flash,
        "grad_checkpoint": grad_checkpoint,
        "batch_size": int(batch),
        "grad_accum": int(accum),
        "num_workers": int(num_workers),
        "compile": compile_mode is not None,
        "compile_mode": compile_mode or "default",
        "matmul_precision": "high",
        "seq_len": int(seq_len),
        "pin_memory": bool(tune.get("pin_memory", True)) and not (
            host_ram is not None and host_ram < 16),
        "tokens_per_micro": int(batch * seq_len),
        "tokenize_chunk_docs": int(feed_chunk),
        "tokenize_pause_s": float(feed_pause),
    }
    sdpa = {"flash": fam["sdp_flash"] and cc[0] >= 8 and cc[0] < 10,
            "mem_efficient": True,
            "cudnn": fam["sdp_cudnn"] and cc[0] >= 8,
            "math": True}
    if cc[0] >= 10:
        sdpa["flash"] = False
        sdpa["cudnn"] = True

    # ------------------------------------------------------- MFU verdict
    if plan["meets_target"]:
        verdict = (f"TARGET MET: projected {plan['mfu']*100:.1f}% >= "
                   f"{mfu_target*100:.0f}% at micro-batch {batch} "
                   f"({batch * seq_len} tokens/micro).")
    elif plan["compute_bound"]:
        verdict = (f"CEILING below target: best projection "
                   f"{plan['mfu']*100:.1f}% vs {mfu_target*100:.0f}% wanted. "
                   "This die is kernel-efficiency-bound for this model size "
                   "(GEMM shapes too small to fill the SMs or pre-tensor-core "
                   "silicon); the only levers left are a bigger variant, fp8 "
                   "hardware, or more accelerators.")
    else:
        verdict = (f"BELOW target: projected {plan['mfu']*100:.1f}% vs "
                   f"{mfu_target*100:.0f}%. Memory bandwidth dominates this "
                   f"(ridge {plan['ridge']:.0f} FLOP/byte); the seeker already "
                   f"maxed the batch that fits ({batch} x {seq_len} tokens). "
                   "More VRAM per token (shard via FSDP) or a smaller seq_len "
                   "would raise arithmetic intensity further.")
    if not plan["meets_target"]:
        warnings.append(verdict)
    else:
        notes.append(verdict)

    cmd = [
        "python", "train.py", f"--variant {vkey}",
        f"--precision {precision}", f"--batch_size {batch}",
        f"--grad_accum {accum}", f"--num_workers {num_workers}",
        f"--seq_len {seq_len}",
        f"--tokenize_chunk_docs {feed_chunk}",
        f"--tokenize_pause_s {feed_pause}",
        "--use_flash_attn" if use_flash else "--no-use_flash_attn",
        "--grad_checkpoint" if grad_checkpoint else "--no-grad_checkpoint",
    ]
    if compile_mode:
        cmd += ["--compile", f"--compile_mode {compile_mode}"]

    return {
        "detected": {
            "name": det.name, "matched_db": matched, "source": det.source,
            "match": det.match, "cc": list(cc), "sms": entry["sms"],
            "vram_gb": vram, "driver": det.driver,
            "gpu_count": det.gpu_count,
            "host_ram_gb": host_ram, "cpu_count": ncpu,
        },
        "family": {"key": family, "label": fam["label"],
                   "note": entry.get("note", "")},
        "tuning": {
            "kernel_eff": float(tune["kernel_eff"]),
            "precision_pref": list(tune.get("precision_pref") or []),
            "compile_mode": compile_mode,
            "ckpt_bias": ckpt_bias,
            "pin_memory": settings["pin_memory"],
            "data_feed": {"workers": int(num_workers),
                          "chunk_docs": int(feed_chunk),
                          "pause_s": float(feed_pause)},
            "rationale": tune.get("ra", ""),
        },
        "peak_tflops": {k: float(v or 0.0) for k, v in peaks.items()},
        "mfu": {"precision": precision, "peak_key": mfu_key,
                "peak_tflops": peak,
                "target": float(mfu_target),
                "projected": float(plan["mfu"]),
                "meets_target": bool(plan["meets_target"]),
                "ridge_flop_per_byte": float(plan["ridge"]),
                "arithmetic_intensity": float(plan["ai"]),
                "wave_factor": float(plan["wave"]),
                "compute_bound": bool(plan["compute_bound"]),
                "verdict": verdict},
        "settings": settings,
        "sdpa_backends": sdpa,
        "env": env,
        "train_cmd": " ".join(cmd),
        "warnings": warnings,
        "notes": notes,
        "memory_model": {
            "static_gb": round(static_gb, 1),
            "per_seq_gb": round(per_seq_gb, 2),
            "usable_gb": round(max(usable, 0.0), 1),
            "cap_ckpt": int(cap_ckpt),
            "cap_nockpt": int(cap_nockpt),
        },
    }


# ===========================================================================
# Application + per-device MFU peaks for the training loop
# ===========================================================================
_PEAK_CACHE: Dict[str, float] = {}


def apply_env_settings(profile: Dict[str, Any]) -> None:
    """Export the profile's env tweaks (call BEFORE CUDA is initialised)."""
    for key, val in profile.get("env", {}).items():
        os.environ[key] = val


def apply_torch_flags(profile: Dict[str, Any]) -> None:
    """Apply backend toggles in-process (matmul precision, SDPA kernels)."""
    try:
        import torch
    except Exception:                                    # pragma: no cover
        return
    try:
        torch.set_float32_matmul_precision(
            profile["settings"].get("matmul_precision", "high"))
        if profile["family"]["key"] not in ("pascal", "volta", "turing"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
        sdp = profile.get("sdpa_backends", {})
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(sdp.get("flash")))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(
                bool(sdp.get("mem_efficient")))
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(bool(sdp.get("cudnn")))
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(bool(sdp.get("math")))
    except Exception as exc:                             # pragma: no cover
        log.warning("apply_torch_flags partially failed: %s", exc)


def peak_tflops_for_device(precision: str = "bf16",
                           device_index: int = 0) -> float:
    """Dense peak TFLOPS of the *visible* GPU for the running precision.

    Powers the trainer's MFU gauge - replaces the old hardcoded 989e12
    (H100-only) denominator so the logged MFU is truthful on any card.
    """
    key = f"{precision}:{device_index}"
    if key in _PEAK_CACHE:
        return _PEAK_CACHE[key]
    peak = 0.0
    try:
        import torch
    except Exception:
        return 0.0
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            entry = match_gpu_db(name)
            if entry is None:
                entry = synthesize_entry(
                    (props.major, props.minor),
                    getattr(props, "multi_processor_count", 80),
                    round(props.total_memory / (1 << 30), 1))
            fam = FAMILIES[entry["family"]]
            pk = {k.lower(): float(v or 0.0)
                  for k, v in entry["peaks"].items()}
            p = precision.strip().lower()
            if p in ("bf16", "fp16"):
                peak = pk.get(p, 0.0)
            elif p == "fp8":
                peak = pk.get("fp8", 0.0) or pk.get("bf16", 0.0)
            elif p == "fp4":
                peak = pk.get("fp4", 0.0) or pk.get("fp8", 0.0) \
                    or pk.get("bf16", 0.0)
            else:                       # fp32 run: TF32 if the HW allows
                peak = pk.get("tf32", 0.0) if fam["tf32"] \
                    else pk.get("fp32", 0.0)
        except Exception:                                # pragma: no cover
            peak = 0.0
    _PEAK_CACHE[key] = peak
    return peak


# ===========================================================================
# Reporting
# ===========================================================================
def pretty_report(profile: Dict[str, Any]) -> str:
    d = profile["detected"]
    f = profile["family"]
    s = profile["settings"]
    p = profile["peak_tflops"]
    m = profile["memory_model"]
    mu = profile["mfu"]
    t = profile.get("tuning", {})

    def _fmt_peak(key: str) -> str:
        val = p.get(key) or 0.0
        return f"{key} {val:.0f}" if val >= 1 else f"{key} -"

    proj = mu.get("projected")
    tgt = mu.get("target", 0.30)
    if proj is not None:
        mfu_line = (f" MFU plan   : projected {proj*100:5.1f}%  vs target "
                    f"{tgt*100:.0f}%   -> {'MET' if mu.get('meets_target') else 'GAP'}"
                    f"   (ridge {mu.get('ridge_flop_per_byte', 0):.0f} FLOP/byte,"
                    f" AI {mu.get('arithmetic_intensity', 0):.0f},"
                    f" wave {mu.get('wave_factor', 0):.2f})")
        feed_line = (f" Data feed  : workers {s['num_workers']}, pacing "
                     f"{s['tokenize_chunk_docs']} docs / {s['tokenize_pause_s']}s nap"
                     f"   pin_memory={s.get('pin_memory', True)}")
    else:
        mfu_line = f" MFU gauge  : precision {mu['precision']} -> {mu['peak_tflops']:.0f} TF peak"
        feed_line = f" Data feed  : workers {s['num_workers']}"

    lines = [
        "=" * 78,
        " OpenMythos GPU auto-tune",
        "=" * 78,
        f" GPU        : {d['name']}"
        + (f"   (matched: {d['matched_db']})" if d["matched_db"] else ""),
        f" Probed via : {d['source']}   x{d['gpu_count']} visible",
        f" Family     : {f['label']}  |  CC {d['cc'][0]}.{d['cc'][1]}"
        f"  |  {d['sms']} SMs  |  {d['vram_gb']:.0f} GB"
        f"  |  driver {d['driver'] or 'n/a'}",
        f" Host       : {d['cpu_count']} CPU threads"
        + (f", {d['host_ram_gb']:.0f} GB RAM" if d["host_ram_gb"] else ""),
        f" Peaks      : " + " | ".join(
            _fmt_peak(k) for k in ("fp32", "tf32", "bf16", "fp16", "fp8",
                                   "fp4") if p.get(k)) + "  (dense TFLOPS)",
        mfu_line,
        "-" * 78,
        " Recommended train.py settings:",
        f"   --precision {s['precision']}"
        f"   --batch_size {s['batch_size']}   --grad_accum {s['grad_accum']}"
        f"   --num_workers {s['num_workers']}",
        f"   {'--use_flash_attn' if s['use_flash_attn'] else '--no-use_flash_attn'}"
        f"   {'--grad_checkpoint' if s['grad_checkpoint'] else '--no-grad_checkpoint'}"
        + (f"   --compile --compile_mode {s['compile_mode']}"
           if s["compile"] else "   (torch.compile off)"),
        f"   --tokenize_chunk_docs {s['tokenize_chunk_docs']}"
        f" --tokenize_pause_s {s['tokenize_pause_s']}",
        feed_line,
        f" Memory plan: static {m['static_gb']} GB + {m['per_seq_gb']} GB/seq,"
        f" usable {m['usable_gb']} GB of {d['vram_gb']:.0f}"
        f"   (cap ckpt {m.get('cap_ckpt', '-')} / no-ckpt {m.get('cap_nockpt', '-')})",
        f" SDPA       : flash={profile['sdpa_backends']['flash']}"
        f" mem_eff={profile['sdpa_backends']['mem_efficient']}"
        f" cudnn={profile['sdpa_backends']['cudnn']}",
    ]
    if t.get("rationale"):
        lines += [f" Arch notes : {t['rationale']}"]
    if t.get("kernel_eff") is not None:
        lines += [f" Kernel eff : {t['kernel_eff']*100:.0f}% of dense peak "
                  f"(large-GEMM ceiling for this die)"]
    lines += [
        "-" * 78,
        " Launch:",
        f"   $ {profile['train_cmd']}",
        f"   (env: " + " ".join(f"{k}={v}" for k, v in
                                  profile["env"].items()) + ")",
    ]
    for w in profile["warnings"]:
        lines += [f" WARNING  : {w}"]
    for n in profile["notes"]:
        lines += [f" note     : {n}"]
    if f.get("note"):
        lines += [f" card note: {f['note']}"]
    lines.append("=" * 78)
    return "\n".join(lines)


# ===========================================================================
# Self-test (runs anywhere - no GPU required)
# ===========================================================================
def self_test(verbose: bool = True) -> Tuple[int, int]:
    """Validate DB integrity + profile invariants for every known card and
    a set of synthetic unknowns. Returns (passed, total)."""
    passed = total = 0

    def check(label: str, cond: bool) -> None:
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            if verbose:
                print(f"  [PASS] {label}")
        else:                                            # pragma: no cover
            print(f"  [FAIL] {label}")

    for entry in GPU_DB:
        name = entry["name"]
        prof = detect_and_build_profile(simulate=name, probe_pkgs=False)
        s = prof["settings"]
        ok = (
            prof["detected"]["matched_db"] == name
            and s["precision"] in ("bf16", "fp16", "fp8", "fp32")
            and isinstance(s["batch_size"], int) and s["batch_size"] >= 1
            and isinstance(s["grad_accum"], int) and s["grad_accum"] >= 1
            and s["num_workers"] >= 1
            and profile_peak_positive(prof)
            and isinstance(prof["env"], dict)
            and all(isinstance(k, str) and isinstance(v, str)
                    for k, v in prof["env"].items())
        )
        check(f"{name:24s} -> {s['precision']:4s} bs={s['batch_size']:<3d} "
              f"acc={s['grad_accum']:<3d} fa={int(s['use_flash_attn'])} "
              f"ckpt={int(s['grad_checkpoint'])} "
              f"peak={prof['mfu']['peak_tflops']:.0f}TF", ok)

        # --- per-card custom-tuning + MFU-plan invariants (the 30% rig) ---
        tune = prof.get("tuning", {})
        mu = prof["mfu"]
        ok2 = (
            name in _TUNE_DB                                  # custom entry
            and 0.0 < float(tune.get("kernel_eff", 0)) <= 1.0
            and isinstance(tune.get("rationale"), str)
            and len(tune.get("rationale", "")) >= 40          # real research
            and isinstance(tune.get("data_feed"), dict)
            and 1 <= int(tune["data_feed"]["workers"]) <= 8
            and int(tune["data_feed"]["chunk_docs"]) >= 30
            and float(tune["data_feed"]["pause_s"]) >= 0.0
            and tune.get("compile_mode") in ("max-autotune",
                                             "max-autotune-no-cudagraphs",
                                             "reduce-overhead", "default",
                                             None)
            and tune.get("ckpt_bias") in ("auto", "on", "off")
            and isinstance(tune.get("pin_memory"), bool)
            and 0.0 <= float(mu.get("projected", -1)) <= 1.0
            and float(mu.get("ridge_flop_per_byte", 0)) > 0
            and float(mu.get("arithmetic_intensity", 0)) > 0
            and 0.45 <= float(mu.get("wave_factor", 0)) <= 1.0
            and isinstance(mu.get("verdict"), str)
            and len(mu.get("verdict", "")) > 20
            and s.get("tokens_per_micro")
                == s["batch_size"] * s.get("seq_len", 2048)
        )
        check(f"{name:24s} -> mfu plan proj={mu.get('projected', 0)*100:4.1f}% "
              f"tgt={mu.get('target', 0)*100:.0f}% "
              f"ridge={mu.get('ridge_flop_per_byte', 0):5.0f} "
              f"{'MET' if mu.get('meets_target') else 'GAP'}", ok2)

    for sim in ("cc=9.0,vram=80,sms=132", "cc=6.1,vram=24,sms=60",
                "cc=12.0,vram=8,sms=20", "cc=10.3,vram=288,sms=160"):
        prof = detect_and_build_profile(simulate=sim, probe_pkgs=False)
        ok = (prof["settings"]["precision"] in ("bf16", "fp16", "fp8", "fp32")
              and profile_peak_positive(prof)
              and prof["detected"]["match"] == "generic")
        check(f"synthetic {sim:28s} -> {prof['settings']['precision']}", ok)

    # precision ladder sanity: pre-Ampere never recommends bf16
    prof = detect_and_build_profile(simulate="T4", probe_pkgs=False)
    check("T4 picks fp16 (tensor cores) not bf16",
          prof["settings"]["precision"] == "fp16")
    prof = detect_and_build_profile(simulate="GTX 1650", probe_pkgs=False)
    check("GTX 1650 picks fp32 (no tensor cores)",
          prof["settings"]["precision"] == "fp32")
    prof = detect_and_build_profile(simulate="B300", probe_pkgs=False)
    check("B300 reports fp4 peak > bf16 peak",
          prof["peak_tflops"]["fp4"] > prof["peak_tflops"]["bf16"])
    prof = detect_and_build_profile(simulate="RTX 5090", probe_pkgs=False)
    check("RTX 5090: 10b cannot fit -> loud warning present",
          any("cannot fit" in w
              for w in detect_and_build_profile(
                  simulate="RTX 5090", probe_pkgs=False,
                  variant="10b")["warnings"]))

    # --- the 30% MFU rig: honest verdicts per scenario -----------------
    prof = detect_and_build_profile(simulate="Tesla P40", probe_pkgs=False)
    check("P40 (no TCs): 30% target honestly reported as GAP",
          not prof["mfu"]["meets_target"]
          and any("CEILING" in w or "BELOW" in w
                  for w in prof["warnings"]))
    prof = detect_and_build_profile(simulate="T4", variant="100m",
                                    seq_len=2048, world_size=2,
                                    probe_pkgs=False)
    # OOM-calibrated (real 2x T4, 2026-08): the unguarded seeker picked
    # no-ckpt batch 28 = 98% of VRAM = OOM on a fresh allocation pattern.
    # With the 85% headroom guard + real usable VRAM + NCCL/graph overhead,
    # the no-ckpt cap is 19; the seeker must stay inside it and still
    # project >= 20% MFU.  19 = int(usable*0.85/per_seq) under current
    # constants - if this pin breaks, re-derive from the memory model.
    check("T4 x2 100m: no-ckpt batch inside 85% headroom guard (OOM-cal)",
          prof["settings"]["grad_checkpoint"] is False
          and prof["settings"]["batch_size"] <= 19
          and prof["mfu"]["projected"] >= 0.20)
    prof = detect_and_build_profile(simulate="A100 80GB", variant="500m",
                                    probe_pkgs=False)
    check("A100 80GB 500m: meets the 30% MFU target",
          prof["mfu"]["meets_target"]
          and prof["mfu"]["projected"] >= 0.30)
    prof = detect_and_build_profile(simulate="H100 SXM", variant="100m",
                                    probe_pkgs=False)
    check("H100 SXM 100m: meets the 30% MFU target",
          prof["mfu"]["meets_target"])
    prof = detect_and_build_profile(simulate="RTX 4090", variant="500m",
                                    probe_pkgs=False)
    check("4090 500m: seeker uses --compile (per-card max-autotune, "
          "graphs stripped)",
          prof["settings"]["compile"]
          and prof["settings"]["compile_mode"] == "max-autotune-no-cudagraphs")
    prof = detect_and_build_profile(simulate="T4", variant="100m",
                                    probe_pkgs=False, mfu_target=0.50)
    check("T4 with 50% target: honest GAP (target respected, not faked)",
          not prof["mfu"]["meets_target"])
    return passed, total


def profile_peak_positive(profile: Dict[str, Any]) -> bool:
    return profile["mfu"]["peak_tflops"] > 0


__all__ = [
    "GPU_DB", "FAMILIES", "Detection",
    "detect_gpu", "detect_and_build_profile", "build_profile",
    "match_gpu_db", "synthesize_entry", "list_known_gpus",
    "apply_env_settings", "apply_torch_flags",
    "peak_tflops_for_device", "pretty_report", "self_test",
]
