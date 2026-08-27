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
    _g("Tesla T4", ["T4"], (7, 5), "turing", 40, 16, 320, 70,
       fp32=8.1, fp16=65, note="70 W cap; fp16 tensor cores are the only win"),
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


def detect_and_build_profile(variant: str = "500m",
                             seq_len: Optional[int] = None,
                             world_size: int = 1,
                             probe_pkgs: bool = True,
                             simulate: Optional[str] = None
                             ) -> Dict[str, Any]:
    det = detect_gpu(simulate=simulate)
    host = probe_host()
    extras = probe_extras() if probe_pkgs else {}
    return build_profile(det, variant=variant, seq_len=seq_len,
                         world_size=world_size, host=host, extras=extras)


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
                  extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    # ----------------------------------------------------------- precision
    has_tc = bool(entry.get("tc", fam["tc_fp16"]))
    if not has_tc:
        precision = "fp32"
        notes.append("No tensor cores on this die: MFU ceiling is raw fp32. "
                     "Expect single-digit utilisation vs modern cards.")
    elif fam["bf16"]:
        precision = "bf16"
    else:                                    # Volta / Turing with fp16 TCs
        precision = "fp16"
        notes.append("Pre-Ampere tensor cores are fp16-only: the trainer "
                     "auto-enables a GradScaler for stable fp16 training.")
    if (precision == "bf16" and fam["fp8"]
            and (cc[0] == 9 or cc[0] == 10)):
        if extras.get("transformer_engine") or extras.get("torchao"):
            precision = "fp8"
            notes.append("FP8 backend present (TE/torchao) and Hopper/"
                         "Blackwell-class silicon: auto-selected for ~2x "
                         "matmul throughput. Pass --precision bf16 to opt out.")
        else:
            notes.append("fp8 supported in silicon; install transformer_"
                         "engine or torchao, then pass --precision fp8.")
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
    static_gb = _NOMINAL_PARAMS_B[vkey] * 16.0 + 0.9
    per_seq_gb = _PER_SEQ_GB[vkey] * (seq_len / 4096.0) ** 1.6
    grad_checkpoint = vram <= 16 or vkey in ("7b", "10b")
    usable = vram * 0.88 - static_gb
    if grad_checkpoint:
        per_seq_gb /= 2.5
    if usable <= 0.6:
        warnings.append(
            f"{vkey}: weights+grads+AdamW states need ~{static_gb:.0f} GB but "
            f"the card has {vram:.0f} GB - single-GPU training cannot fit. "
            "Use torchrun --nproc_per_node>=2 with --dist_strategy fsdp, or "
            "drop to a smaller variant.")
        grad_checkpoint = True
        batch = 1
    else:
        batch = max(1, min(64, int(usable / max(per_seq_gb, 0.4))))
    target = _TARGET_SEQS[vkey]
    accum = max(1, min(256, round(target / max(batch, 1))))

    if vkey in ("7b", "10b") and not grad_checkpoint:
        grad_checkpoint = True

    # ------------------------------------------------------------ workers
    ncpu = int(host.get("cpu_count") or 4)
    num_workers = max(2, min(8, ncpu // 2))
    if cc[0] <= 7:
        num_workers = min(num_workers, 4)

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
    compile_mode = None
    if fam["compile"]:
        compile_mode = ("max-autotune"
                        if family in ("hopper", "bw_dc")
                        or cc == (8, 0) else "default")
        if vram < 12:
            compile_mode = None
            notes.append("Small VRAM: torch.compile's extra graph buffers "
                         "outweigh its speedup - left off.")

    peaks = entry["peaks"]
    mfu_key = {"fp32": "tf32" if fam["tf32"] else "fp32",
               "bf16": "bf16", "fp16": "fp16",
               "fp8": "fp8", "fp4": "fp4"}[precision]
    peak = float(peaks.get(mfu_key) or peaks.get("bf16") or 0.0)

    settings = {
        "precision": precision,
        "use_flash_attn": use_flash,
        "grad_checkpoint": bool(grad_checkpoint),
        "batch_size": batch,
        "grad_accum": accum,
        "num_workers": num_workers,
        "compile": compile_mode is not None,
        "compile_mode": compile_mode or "default",
        "matmul_precision": "high",
    }
    sdpa = {"flash": fam["sdp_flash"] and cc[0] >= 8 and cc[0] < 10,
            "mem_efficient": True,
            "cudnn": fam["sdp_cudnn"] and cc[0] >= 8,
            "math": True}
    if cc[0] >= 10:
        sdpa["flash"] = False
        sdpa["cudnn"] = True

    cmd = [
        "python", "train.py", f"--variant {vkey}",
        f"--precision {precision}", f"--batch_size {batch}",
        f"--grad_accum {accum}", f"--num_workers {num_workers}",
        f"--seq_len {seq_len}",
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
        "peak_tflops": {k: float(v or 0.0) for k, v in peaks.items()},
        "mfu": {"precision": precision, "peak_key": mfu_key,
                "peak_tflops": peak},
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

    def _fmt_peak(key: str) -> str:
        val = p.get(key) or 0.0
        return f"{key} {val:.0f}" if val >= 1 else f"{key} -"

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
        f" MFU gauge  : precision {profile['mfu']['precision']} -> "
        f"{profile['mfu']['peak_tflops']:.0f} TF peak",
        "-" * 78,
        " Recommended train.py settings:",
        f"   --precision {s['precision']}"
        f"   --batch_size {s['batch_size']}   --grad_accum {s['grad_accum']}"
        f"   --num_workers {s['num_workers']}",
        f"   {'--use_flash_attn' if s['use_flash_attn'] else '--no-use_flash_attn'}"
        f"   {'--grad_checkpoint' if s['grad_checkpoint'] else '--no-grad_checkpoint'}"
        + (f"   --compile --compile_mode {s['compile_mode']}"
           if s["compile"] else "   (torch.compile off)"),
        f" Memory plan: static {m['static_gb']} GB + {m['per_seq_gb']} GB/seq,"
        f" usable {m['usable_gb']} GB of {d['vram_gb']:.0f}",
        f" SDPA       : flash={profile['sdpa_backends']['flash']}"
        f" mem_eff={profile['sdpa_backends']['mem_efficient']}"
        f" cudnn={profile['sdpa_backends']['cudnn']}",
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
