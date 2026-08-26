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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", choices=["bf16", "fp8", "fp4", "fp32"],
                   default="bf16")
    p.add_argument("--loop_iters", type=int, default=8,
                   help="recurrent depth T executed per forward pass")
    p.add_argument("--use_flash_attn", action=argparse.BooleanOptionalAction,
                   default=True, help="prefer fused FlashAttention kernels")
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
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer_docs", type=int, default=2048,
                   help="streaming reshuffle reservoir size in documents; "
                        "lower on RAM-constrained machines")
    p.add_argument("--low_ram", action="store_true",
                   help="constant-RAM profile: caps shuffle buffer at 512 "
                        "docs and disables pinned-memory staging")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dist_strategy", choices=["auto", "ddp", "fsdp", "none"],
                   default="auto")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="recompute loop steps to save VRAM")
    p.add_argument("--log_interval", type=int, default=10)
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

    # ---- environment bootstrap ---------------------------------------------
    torch.set_float32_matmul_precision("high")     # TF32 paths on Ampere+
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dist = setup_distributed()
    device = dist.device
    is_main = dist.is_main
    set_seed_everywhere(args.seed + (dist.rank * 100003))

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
        **overrides,
    )

    model = build_checkpointable_model(args, cfg)
    prepare_model_for_precision(model, args.precision)

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

    for step in range(start_step, args.max_steps):
        lr_now = scheduler(step)
        micro_losses = []
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
            scaled.backward()
            micro_losses.append(info["ce"])
            tokens_seen += batch_x.numel()
            del logits, loss_total, batch_x, batch_y

            if micro == args.grad_accum - 1:
                torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
                optimizer.step()
                scheduler(step + 1)
                optimizer.zero_grad(set_to_none=True)
                stepped_this_round = True

        if not stepped_this_round:
            continue

        tokens_per_sec = tokens_seen / max(time.time() - t0, 1e-6)
        ce_mean = sum(micro_losses) / max(len(micro_losses), 1)

        if is_main and logger and step % args.log_interval == 0:
            mfu_est = (
                3.0 * flops_fwd * tokens_per_sec
                / (989e12 * max(dist.world_size, 1))   # dense-bf16 peak assumption
            ) if device.startswith("cuda") else 0.0
            payload = {
                "loss/ce": ce_mean,
                "loss/ce_last": micro_losses[-1] if micro_losses else ce_mean,
                "train/lr": lr_now,
                "throughput/tokens_per_s": tokens_per_sec,
                "mfu/estimate": mfu_est,
                "moe/router_entropy": getattr(
                    raw.recurrent_stack[0].moe, "last_router_entropy", 0.0
                ),
            }
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

        if (step + 1) % args.checkpoint_interval == 0 and is_main:
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
