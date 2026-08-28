#!/usr/bin/env python3
"""OpenMythos training entrypoint.

Single GPU::

    python train.py --variant 1b --precision bf16 --batch_size 8

Multi-GPU (DDP / FSDP auto-selected)::

    torchrun --nproc_per_node=8 train.py --variant 10b --precision fp8

The script streams HuggingFace FineWeb-Edu continuously, applies the
requested precision backend, wraps the model for distributed execution,
trains with AdamW + cosine/warmup scheduling, and checkpoints periodically.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openmythos.attention import flash_attention_backend                    # noqa: E402
from openmythos.config import KNOWN_VARIANTS, MythosConfig                  # noqa: E402
from openmythos.dataset import get_fineweb_dataloader                       # noqa: E402
from openmythos.gpu_profile import peak_tflops_for_device                   # noqa: E402
from openmythos.model import OpenMythosForCausalLM                          # noqa: E402
from openmythos.precision import (                                          # noqa: E402
    describe_device_precision_hardware,
    get_autocast_context,
    prepare_model_for_precision,
)
from openmythos.utils import (                                              # noqa: E402
    CosineAnnealingWithWarmup,
    LoggerWrapper,
    count_parameters,
    estimate_flops_per_token,
    setup_distributed,
    set_seed_everywhere,
    teardown_distributed,
    unwrap_model,
    wrap_model,
)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train OpenMythos Recurrent-Depth Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- spec-mandated core flags -----------------------------------------
    p.add_argument("--variant", choices=KNOWN_VARIANTS, default="500m",
                   help="model size preset to instantiate")
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--auto_tune", action="store_true",
                   help="detect the local GPU and derive MFU-optimal "
                        "precision/batch/workers settings; explicit flags "
                        "always win over tuned values")
    p.add_argument("--batch_size", type=int, default=None,
                   help="micro-batch per GPU (default: 4, or auto_tuned)")
    p.add_argument("--grad_accum", type=int, default=None,
                   help="accumulation steps (default: 8, or auto_tuned)")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision",
                   choices=["bf16", "fp16", "fp8", "fp4", "fp32"],
                   default=None,
                   help="bf16 (default / Ampere+) | fp16 (pre-Ampere tensor "
                        "cores, GradScaler auto-enabled) | fp8 | fp4 | fp32")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="torch.compile the model (default: auto_tuned per "
                        "GPU card; explicit --compile/--no-compile wins)")
    p.add_argument("--compile_mode", default=None,
                   choices=["default", "max-autotune", "reduce-overhead"],
                   help="torch.compile mode (default: the detected card's "
                        "per-GPU tuned mode)")
    p.add_argument("--loop_iters", type=int, default=8,
                   help="recurrent depth T executed per forward pass")
    p.add_argument("--use_flash_attn", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="prefer fused FlashAttention kernels (default: True, "
                        "or auto_tuned off pre-Ampere / Blackwell)")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--wandb_project", default="openmythos-fineweb")

    # --- production extras --------------------------------------------------
    p.add_argument("--tokenizer_name", default="gpt2",
                   help="tiktoken encoding: gpt2 | cl100k_base | char256")
    p.add_argument("--attn_type", choices=["gqa", "mla"], default=None,
                   help="override the preset attention backbone")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=256)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_loss_coeff", type=float, default=0.01)
    p.add_argument("--z_loss_coeff", type=float, default=1e-3,
                   help="router logit-magnitude (ST-MoE z) loss weight; "
                        "damps top-k routing flips, a classic source of "
                        "transient loss spikes in MoE training")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=None,
                   help="dataloader workers (default: 4, or auto_tuned)")
    p.add_argument("--shuffle_buffer_docs", type=int, default=2048,
                   help="streaming reshuffle reservoir size in documents; "
                        "lower on RAM-constrained machines")
    p.add_argument("--low_ram", action="store_true",
                   help="constant-RAM profile: caps shuffle buffer at 512 "
                        "docs and disables pinned-memory staging")
    p.add_argument("--tokenize_chunk_docs", type=int, default=None,
                   help="download-pacing gate: after every N raw documents "
                        "the stream tokenises/packs the batch fully before "
                        "resuming the parquet download (default: the "
                        "detected card's tuned data-feed value, else 30)")
    p.add_argument("--tokenize_pause_s", type=float, default=None,
                   help="timer nap (seconds) after each paced chunk "
                        "(default: the detected card's tuned data-feed "
                        "value, else 0.05)")
    p.add_argument("--data_mode", choices=["native", "stream"], default="native",
                   help="native: huggingface_hub shard download + local "
                        "pyarrow reads (timeout-proof); stream: legacy live "
                        "load_dataset(streaming=True) HTTP reader")
    p.add_argument("--max_parquet_shards", type=int, default=64,
                   help="disk cap for native mode: how many remote parquet "
                        "shards (~100-300 MB each) to cache")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dist_strategy", choices=["auto", "ddp", "fsdp", "none"],
                   default="auto")
    p.add_argument("--grad_checkpoint",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="recompute loop steps to save VRAM (default: off, "
                        "or auto_tuned on for small cards)")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--scaler_init_scale", type=float, default=65536.0,
                   help="fp16 GradScaler initial loss scale")
    p.add_argument("--scaler_growth_interval", type=int, default=2000,
                   help="fp16 GradScaler: steps between 2x scale growth; "
                        "raise (e.g. 4000) to reduce overflow-skip churn")
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=16)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume", default=None,
                   help="path to checkpoint .pt file to resume from")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--tensorboard_dir", default="./tb_logs")
    p.add_argument("--disable_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def build_checkpointable_model(args, cfg):
    model = OpenMythosForCausalLM(cfg)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(True)
    return model


@torch.no_grad()
def run_evaluation(model, eval_loader, autocast_ctx_factory, device, iters):
    was_training = model.training
    model.eval()
    losses = []
    it = iter(eval_loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx_factory():
            _, _, info = model(x, y)
        losses.append(info["ce"])
        del x, y
    if was_training:
        model.train()
    mean_ce = sum(losses) / max(len(losses), 1)
    return mean_ce, math.exp(min(mean_ce, 20.0))


def save_checkpoint(path: Path, raw_model, optimizer, scheduler, dist_state,
                    step: int, best_val: float, extra: dict) -> None:
    payload = {
        "step": step,
        "best_val": best_val,
        "model": unwrap_model(raw_model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        },
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic swap survives crashes mid-write


def try_load_checkpoint(path: Path, model, optimizer, scheduler):
    """Restore trainer state.

    Loads through the distributed wrapper first (correct for FSDP whose
    ``use_orig_params`` state includes sharded views), falling back to the
    bare nn.Module for plain checkpoints saved by earlier runs.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    restored = False
    try:
        model.load_state_dict(blob["model"])
        restored = True
    except Exception as exc:
        logging.getLogger("openmythos.train").info(
            "wrapper-level restore failed (%s); trying unwrapped keys.", exc
        )
    if not restored:
        unwrap_model(model).load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    if blob.get("rng", {}).get("torch") is not None:
        torch.set_rng_state(blob["rng"]["torch"].cpu().to(torch.uint8))
    if (torch.cuda.is_available()
            and blob.get("rng", {}).get("cuda") is not None):
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8)
                                          for s in blob["rng"]["cuda"]])
        except Exception as exc:                              # pragma: no cover
            logging.getLogger("openmythos.train").warning(
                "CUDA RNG restore skipped (%s)", exc)
    return int(blob["step"]), float(blob.get("best_val", float("inf")))


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = logging.getLogger("openmythos.train")

    # The per-request httpx INFO lines (302 redirects, xet bridge handshakes)
    # are pure noise at training scale; hub download progress stays visible.
    for noisy in ("httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- GPU auto-tune (env exports must precede CUDA initialisation) ------
    auto_profile = None
    if args.auto_tune:
        from openmythos.gpu_profile import (
            apply_env_settings,
            apply_torch_flags,
            detect_and_build_profile,
            pretty_report,
        )

        world_guess = int(os.environ.get("WORLD_SIZE", "1") or 1)
        try:
            auto_profile = detect_and_build_profile(
                variant=args.variant, seq_len=args.seq_len,
                world_size=world_guess,
            )
            if auto_profile["detected"]["source"] == "none":
                log.warning(
                    "--auto_tune: no NVIDIA GPU visible; keeping manual "
                    "defaults (tuning is a no-op on CPU-only hosts)."
                )
                auto_profile = None
            else:
                apply_env_settings(auto_profile)   # before any CUDA context
        except Exception as exc:                   # noqa: BLE001
            log.warning("--auto_tune probe failed (%s); using defaults.", exc)
            auto_profile = None

    # Explicit flags win; auto-tuned values second; legacy defaults last.
    tuned = auto_profile["settings"] if auto_profile else {}

    def _resolve(value, tuned_val, fallback):
        if value is not None:
            return value
        return tuned_val if tuned_val is not None else fallback

    args.precision = _resolve(args.precision, tuned.get("precision"), "bf16")
    args.batch_size = int(
        _resolve(args.batch_size, tuned.get("batch_size"), 4))
    args.grad_accum = int(
        _resolve(args.grad_accum, tuned.get("grad_accum"), 8))
    args.num_workers = int(
        _resolve(args.num_workers, tuned.get("num_workers"), 4))
    args.use_flash_attn = bool(
        _resolve(args.use_flash_attn, tuned.get("use_flash_attn"), True))
    args.grad_checkpoint = bool(
        _resolve(args.grad_checkpoint, tuned.get("grad_checkpoint"), False))
    args.compile = bool(
        _resolve(args.compile, tuned.get("compile"), False))
    args.compile_mode = str(
        _resolve(args.compile_mode, tuned.get("compile_mode"), "default"))
    args.tokenize_chunk_docs = int(
        _resolve(args.tokenize_chunk_docs,
                 tuned.get("tokenize_chunk_docs"), 30))
    args.tokenize_pause_s = float(
        _resolve(args.tokenize_pause_s,
                 tuned.get("tokenize_pause_s"), 0.05))

    # ---- environment bootstrap ---------------------------------------------
    torch.set_float32_matmul_precision("high")     # TF32 paths on Ampere+
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dist = setup_distributed()
    device = dist.device
    is_main = dist.is_main
    set_seed_everywhere(args.seed + (dist.rank * 100003))

    if auto_profile is not None:
        apply_torch_flags(auto_profile)   # SDPA kernel priorities per GPU
        if is_main:
            log.info("\n%s", pretty_report(auto_profile))
            _mp = auto_profile.get("mfu", {})
            if _mp.get("projected") is not None:
                log.info(
                    "MFU plan: projected %.1f%% vs %.0f%% target -> %s "
                    "(batch %d x seq %d, %d tokens/micro)",
                    _mp["projected"] * 100, _mp.get("target", 0.30) * 100,
                    "TARGET MET" if _mp.get("meets_target")
                    else "gap - see verdict in banner",
                    auto_profile["settings"]["batch_size"],
                    auto_profile["settings"]["seq_len"],
                    auto_profile["settings"]["tokens_per_micro"],
                )

    log.info("Hardware: %s", describe_device_precision_hardware())
    log.info("Attention kernels: %s backend selected",
             flash_attention_backend())

    # ---- data (built FIRST so vocab matches the tokenizer) ------------------
    train_loader, meta = get_fineweb_dataloader(
        dataset_name=args.dataset_name,
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer_name,
        seed=args.seed + dist.rank,
        shuffle_buffer_docs=args.shuffle_buffer_docs,
        low_ram_profile=args.low_ram,
        tokenize_chunk_docs=args.tokenize_chunk_docs,
        tokenize_pause_s=args.tokenize_pause_s,
        data_mode=args.data_mode,
        max_parquet_shards=args.max_parquet_shards,
    )
    eval_loader, eval_meta = get_fineweb_dataloader(
        dataset_name=args.dataset_name,
        split="train",
        batch_size=max(1, args.batch_size),
        seq_len=args.seq_len,
        num_workers=0,          # main-process streaming: no extra worker RSS
                                # and no process pile-up across repeated evals
        tokenizer_name=args.tokenizer_name,
        seed=args.seed + 777 + dist.rank,
        shuffle_buffer_docs=args.shuffle_buffer_docs,
        low_ram_profile=True,   # eval never needs deep shuffling
        tokenize_chunk_docs=args.tokenize_chunk_docs,
        tokenize_pause_s=args.tokenize_pause_s,
        data_mode=args.data_mode,
        max_parquet_shards=args.max_parquet_shards,
    )
    if meta.demo_mode or eval_meta.demo_mode:
        log.warning("=" * 78)
        log.warning("DEMO MODE: synthetic offline corpus active "
                    "(no network / HF unavailable). Metrics are NOT meaningful.")
        log.warning("=" * 78)

    # ---- model ----------------------------------------------------------------
    overrides = {}
    if args.attn_type:
        overrides["attn_type"] = args.attn_type
    cfg = MythosConfig.from_variant(
        args.variant,
        vocab_size=meta.vocab_size,
        max_seq_len=args.seq_len,
        max_loop_iters=args.loop_iters,
        use_flash_attn=args.use_flash_attn,
        dropout=args.dropout,
        aux_loss_coeff=args.aux_loss_coeff,
        z_loss_coeff=args.z_loss_coeff,
        **overrides,
    )

    model = build_checkpointable_model(args, cfg)
    prepare_model_for_precision(model, args.precision)
    if args.compile:
        if args.dist_strategy == "fsdp":
            log.warning("--compile skipped under --dist_strategy fsdp "
                        "(compile x FSDP wrap order is fragile).")
        else:
            log.info("torch.compile(mode=%s) compiling model...",
                     args.compile_mode)
            model = torch.compile(model, mode=args.compile_mode)

    stats = count_parameters(model, cfg)
    flops_fwd = estimate_flops_per_token(cfg, seq_len=args.seq_len)
    if is_main:
        log.info("\n%s", cfg.describe())
        log.info(
            "Parameters: total %.2fM | active/token %.2fM | routed pool %.2fM"
            " | non-expert %.2fM",
            stats["total_parameters"] / 1e6,
            stats["active_parameters_per_token"] / 1e6,
            stats["routed_expert_parameters"] / 1e6,
            stats["non_expert_parameters"] / 1e6,
        )
        log.info("Approx forward FLOPs/token: %.3f GFLOPs", flops_fwd / 1e9)

    # ---- distributed wrapping ---------------------------------------------------
    wrapped = wrap_model(model, dist, strategy=args.dist_strategy)
    raw = unwrap_model(wrapped)

    optimizer = raw.configure_optimizers(
        weight_decay=args.weight_decay,
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    accum_scale = max(args.grad_accum * max(dist.world_size, 1), 1)
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=max(args.max_steps, 1),
        min_lr_ratio=args.min_lr_ratio,
    )

    precision_ctx_factory = lambda: get_autocast_context(args.precision)  # noqa: E731

    # fp16 runs (pre-Ampere tensor cores) need loss scaling to keep small
    # gradients alive; bf16/fp8/fp4 paths scale-free and leave it disabled.
    use_scaler = args.precision == "fp16" and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
        init_scale=args.scaler_init_scale,
        growth_interval=args.scaler_growth_interval,
    )
    if use_scaler:
        log.info("GradScaler enabled for fp16 training (dynamic loss scale, "
                 "init %.0f, growth every %d steps).",
                 args.scaler_init_scale, args.scaler_growth_interval)

    # Truthful MFU denominator: dense peak of THIS card for THIS precision,
    # replacing the old hardcoded H100 constant.
    peak_tflops = (
        peak_tflops_for_device(args.precision)
        if str(device).startswith("cuda") else 0.0
    )
    if is_main:
        if peak_tflops > 0:
            log.info("MFU gauge: using %.0f TFLOPS dense peak for %s.",
                     peak_tflops, args.precision)
        else:
            log.info("MFU gauge: peak unknown for this GPU; mfu/estimate "
                     "will read 0.")

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            candidate = ckpt_dir / f"{args.variant}-latest.pt"
            if candidate.exists():
                resume_path = candidate
        if resume_path.exists():
            start_step, best_val = try_load_checkpoint(
                resume_path, wrapped, optimizer, scheduler
            )
            log.info("Resumed from %s at global step %d.", resume_path, start_step)
        else:
            log.warning("--resume given but no checkpoint found at %s or %s.",
                        args.resume, ckpt_dir / f"{args.variant}-latest.pt")

    logger = LoggerWrapper(
        project=args.wandb_project,
        run_name=args.wandb_run_name
        or f"{args.variant}-{args.precision}-{time.strftime('%Y%m%d-%H%M%S')}",
        config_payload={**vars(args), "config": cfg.to_dict()},
        enabled=is_main and not args.disable_wandb,
        tensorboard_dir=args.tensorboard_dir,
    ) if is_main else None

    # ---- training loop ---------------------------------------------------------
    model_device_param = next(raw.parameters()).device
    wrapped.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    tokens_seen = 0
    skipped_steps = 0          # fp16 overflow skips (update not applied)
    ema_ce: float = float("nan")   # spike detector baseline

    for step in range(start_step, args.max_steps):
        lr_now = scheduler(step)
        micro_losses = []
        last_info: dict = {}
        grad_norm: float = 0.0
        stepped_this_round = False

        for micro in range(args.grad_accum):
            try:
                batch_x, batch_y = next(data_iter)
            except StopIteration:                      # defensive; stream is endless
                data_iter = iter(train_loader)
                continue
            except (ConnectionError, RuntimeError) as stream_err:
                log.warning("Stream hiccup (%s); recreating iterator.", stream_err)
                time.sleep(min(60.0, 5.0 * (micro + 1)))
                data_iter = iter(train_loader)
                continue

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with precision_ctx_factory():
                logits, loss_total, info = wrapped(batch_x, batch_y)
            scaled = loss_total / accum_scale
            if use_scaler:
                scaled = scaler.scale(scaled)
            scaled.backward()
            micro_losses.append(info["ce"])
            last_info = info
            tokens_seen += batch_x.numel()
            del logits, loss_total, batch_x, batch_y

            if micro == args.grad_accum - 1:
                pre_scale = scaler.get_scale() if use_scaler else 0.0
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    raw.parameters(), args.grad_clip))
                stepped_ok = True
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < pre_scale:
                        # Overflow: the update was NOT applied.  Do not burn
                        # an LR-schedule step on a no-op.
                        stepped_ok = False
                        skipped_steps += 1
                        if skipped_steps <= 5 or skipped_steps % 50 == 0:
                            log.warning(
                                "fp16 overflow @step %d: optimizer step "
                                "skipped, loss scale -> %.0f (%d skips "
                                "total). If this repeats every few hundred "
                                "steps, raise --scaler_growth_interval.",
                                step, scaler.get_scale(), skipped_steps)
                else:
                    optimizer.step()
                if stepped_ok:
                    scheduler(step + 1)
                optimizer.zero_grad(set_to_none=True)
                stepped_this_round = True

        if not stepped_this_round:
            continue

        tokens_per_sec = tokens_seen / max(time.time() - t0, 1e-6)
        ce_mean = sum(micro_losses) / max(len(micro_losses), 1)

        if is_main and logger and step % args.log_interval == 0:
            # ---- spike detector: compare against a slow EMA of the CE ----
            prev_ema = ema_ce
            ema_ce = ce_mean if math.isnan(ema_ce) else 0.9 * ema_ce + 0.1 * ce_mean
            if not math.isnan(prev_ema) and ce_mean > prev_ema + 0.75:
                log.warning(
                    "loss spike @step %d: ce=%.3f (ema=%.3f) | "
                    "grad_norm=%.2f | router_entropy=%.3f | "
                    "aux=%.4f z=%.4f%s -- a single spike that recovers by "
                    "itself is usually one hard batch; recurring spikes "
                    "with grad_norm near clip mean fp16 scale churn.",
                    step, ce_mean, prev_ema, grad_norm,
                    float(getattr(raw.recurrent_stack[0].moe,
                                  "last_router_entropy", 0.0) or 0.0),
                    last_info.get("aux", 0.0), last_info.get("z", 0.0),
                    f" | loss_scale={scaler.get_scale():.0f}"
                    if use_scaler else "",
                )
            mfu_est = (
                3.0 * flops_fwd * tokens_per_sec
                / (peak_tflops * 1e12 * max(dist.world_size, 1))
            ) if peak_tflops > 0 else 0.0
            payload = {
                "loss/ce": ce_mean,
                "loss/ce_last": micro_losses[-1] if micro_losses else ce_mean,
                "loss/aux": last_info.get("aux", 0.0),
                "loss/z": last_info.get("z", 0.0),
                "train/lr": lr_now,
                "train/grad_norm": grad_norm,
                "train/skipped_steps": skipped_steps,
                "throughput/tokens_per_s": tokens_per_sec,
                "mfu/estimate": mfu_est,
                "moe/router_entropy": getattr(
                    raw.recurrent_stack[0].moe, "last_router_entropy", 0.0
                ),
            }
            if use_scaler:
                payload["train/loss_scale"] = scaler.get_scale()
            logger.log(payload, step=step)

        should_eval = (
            args.eval_interval > 0
            and (step + 1) % args.eval_interval == 0
        )
        if should_eval and is_main:
            val_ce, val_ppl = run_evaluation(
                raw, eval_loader, precision_ctx_factory,
                str(model_device_param), args.eval_iters,
            )
            if logger:
                logger.log({"val/loss": val_ce, "val/perplexity": val_ppl},
                           step=step)
            log.info("validation @%d: loss=%.4f ppl=%.2f", step, val_ce, val_ppl)
            best_val = min(best_val, val_ce)
            save_checkpoint(
                ckpt_dir / f"{args.variant}-best.pt", wrapped, optimizer,
                scheduler, dist, step + 1, best_val,
                {"config": cfg.to_dict(), "args": vars(args)},
            )

        if (args.checkpoint_interval > 0
                and (step + 1) % args.checkpoint_interval == 0 and is_main):
            save_checkpoint(
                ckpt_dir / f"{args.variant}-latest.pt", wrapped, optimizer,
                scheduler, dist, step + 1, best_val,
                {"config": cfg.to_dict(), "args": vars(args)},
            )
            if not should_eval:
                log.info("checkpoint saved @ step %d", step + 1)

    elapsed = time.time() - t0
    if is_main:
        log.info(
            "Training finished: %d steps in %.1f min (%.0f tok/s avg).",
            args.max_steps, elapsed / 60.0,
            tokens_seen / max(elapsed, 1e-6),
        )
        if logger:
            save_checkpoint(
                ckpt_dir / f"{args.variant}-latest.pt", wrapped, optimizer,
                scheduler, dist, args.max_steps, best_val,
                {"config": cfg.to_dict(), "args": vars(args)},
            )
            logger.log({"final/train_seconds": elapsed}, step=args.max_steps)
            logger.finish()
    teardown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
