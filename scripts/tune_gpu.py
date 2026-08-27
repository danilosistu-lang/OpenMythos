#!/usr/bin/env python3
"""GPU detection & MFU tuning CLI for OpenMythos.

Examples
--------
    python scripts/tune_gpu.py                     # pretty report for this host
    python scripts/tune_gpu.py --json              # machine-readable profile
    python scripts/tune_gpu.py --env               # shell: eval $(... --env)
    python scripts/tune_gpu.py --simulate B300     # profile any known card
    python scripts/tune_gpu.py --simulate "cc=9.0,vram=80,sms=132"
    python scripts/tune_gpu.py --self-test         # validate the whole DB
    python scripts/tune_gpu.py --bench             # measure real GEMM TFLOPS
    python scripts/tune_gpu.py --list              # dump the GPU database
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect the GPU and emit MFU-optimal training settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="500m",
                   help="model size the settings should target")
    p.add_argument("--seq_len", type=int, default=None,
                   help="training sequence length (default: variant preset)")
    p.add_argument("--world_size", type=int,
                   default=int(__import__("os").environ.get("WORLD_SIZE", "1")
                               or 1),
                   help="number of processes that will run (torchrun)")
    p.add_argument("--json", action="store_true", help="emit JSON profile")
    p.add_argument("--env", action="store_true",
                   help="emit 'export KEY=VAL' lines for shell eval")
    p.add_argument("--quiet", action="store_true",
                   help="suppress notes/warnings chatter in --env mode")
    p.add_argument("--simulate", default=None, metavar="NAME_OR_SPEC",
                   help="profile a known card name or a "
                        "'cc=M.m,vram=GB,sms=N' spec instead of this host")
    p.add_argument("--list", action="store_true",
                   help="list every GPU in the database and exit")
    p.add_argument("--self-test", action="store_true",
                   help="validate DB + profiles for all cards (CPU safe)")
    p.add_argument("--bench", action="store_true",
                   help="measure achievable GEMM TFLOPS on the real GPU")
    p.add_argument("--bench_m", type=int, default=8192,
                   help="GEMM matrix edge for --bench")
    return p


def _cmd_list() -> int:
    from openmythos.gpu_profile import GPU_DB, FAMILIES
    print(f"{'GPU':24s} {'CC':>5s} {'SMs':>4s} {'GB':>5s} {'GB/s':>5s} "
          f"{'W':>4s}  {'family':<12s} {'bf16':>5s} {'fp16':>5s} {'fp8':>5s} "
          f"{'fp4':>6s}  note")
    for e in GPU_DB:
        pk = e["peaks"]
        print(f"{e['name']:24s} {e['cc'][0]}.{e['cc'][1]:<3d} "
              f"{e['sms']:>4d} {e['vram']:>5.0f} {e['bw']:>5d} "
              f"{e['tdp']:>4d}  {e['family']:<12s} "
              f"{pk['bf16'] or 0:>5.0f} {pk['fp16'] or 0:>5.0f} "
              f"{pk['fp8'] or 0:>5.0f} {pk['fp4'] or 0:>6.0f}  "
              f"{e.get('note', '')}")
    return 0


def _cmd_bench(m: int) -> int:
    try:
        import torch
    except Exception:
        print("torch not installed - benchmark unavailable.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("No CUDA device visible - nothing to benchmark.", file=sys.stderr)
        return 2
    name = torch.cuda.get_device_name(0)
    print(f"Benchmarking {name} (matmul {m}x{m}, bf16 + fp16)...")
    results = {}
    for dtype, label in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        try:
            a = torch.randn(m, m, device="cuda", dtype=dtype)
            b = torch.randn(m, m, device="cuda", dtype=dtype)
            for _ in range(10):
                a @ b
            torch.cuda.synchronize()
            t0 = time.time()
            iters = 0
            while time.time() - t0 < 3.0:
                for _ in range(10):
                    a @ b
                torch.cuda.synchronize()
                iters += 10
            dt = time.time() - t0
            tf = 2 * m ** 3 * iters / dt / 1e12
            results[label] = tf
            print(f"  {label}: {tf:.1f} TFLOPS dense")
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:                         # noqa: BLE001
            print(f"  {label}: failed ({exc})")
    from openmythos.gpu_profile import peak_tflops_for_device
    peak = peak_tflops_for_device("bf16")
    if peak > 0 and results.get("bf16"):
        print(f"  bf16 efficiency vs datasheet: "
              f"{100.0 * results['bf16'] / peak:.0f}% "
              f"(measured {results['bf16']:.0f} / peak {peak:.0f} TF)")
        print("  A well-tuned full training run typically reaches 35-55% MFU "
              "of the DATASHEET peak; use the train.py MFU log line to track it.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return _cmd_list()
    if args.self_test:
        from openmythos.gpu_profile import self_test
        passed, total = self_test()
        print(f"SELF-TEST: {passed}/{total} PASS")
        return 0 if passed == total else 1

    from openmythos.gpu_profile import (apply_env_settings,  # noqa: F401
                                        detect_and_build_profile,
                                        pretty_report)
    profile = detect_and_build_profile(
        variant=args.variant, seq_len=args.seq_len,
        world_size=args.world_size, probe_pkgs=not args.simulate,
        simulate=args.simulate,
    )

    if args.json:
        print(json.dumps(profile, indent=2))
        return 0
    if args.env:
        for key, val in profile["env"].items():
            print(f"export {key}={val}")
        if not args.quiet:
            import shlex
            print(f"export OPENMYTHOS_TUNED_CMD={shlex.quote(profile['train_cmd'])}")
        return 0
    print(pretty_report(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
