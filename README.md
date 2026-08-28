# OpenMythos — Recurrent-Depth Transformer (RDT)

**A production-ready, runnable implementation of Recurrent-Depth Transformers
targeting NVIDIA Blackwell (B200 / GB200) and Hopper (H100 / H200) fleets.**

OpenMythos trains transformer models that *think for longer at inference* not by
adding layers, but by **executing a weight-shared recurrent core for a
configurable number of iterations** — with spectrally-guaranteed stability,
sparse Mixture-of-Experts economics, per-loop-step LoRA specialisation, and
native BF16 / FP8 / NVFP4 precision paths.

```
        ┌────────────┐   ┌───────────────────────────────┐   ┌──────────┐
tokens→ │  PRELUDE   │ → │  RECURRENT CORE (loop × T)    │ → │   CODA   │→ logits
  x     │ blocks ×P  │e  │  h ← A·h + B·e + LoRA_t(...)  │h_T│ blocks×C │
        │ (run once) │   │  + Block(h) weights SHARED    │   │(run once)│
        └────────────┘   └───────────────────────────────┘   └──────────┘
                                  loop t = 0 … T−1
```

---

## Table of contents

1. [Architecture](#architecture)
2. [Theoretical background](#theoretical-background)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Model variants](#model-variants)
7. [Precision scaling guide (BF16 / FP16 / FP8 / NVFP4)](#precision-scaling-guide)
8. [GPU auto-tune — MFU profiles for every card](#gpu-auto-tune--mfu-profiles-for-every-card)
9. [Training stability — loss spikes, fp16 skips, router health](#training-stability--loss-spikes-fp16-skips-router-health)
10. [Dataset streaming](#dataset-streaming)
11. [Distributed training](#distributed-training)
12. [Checkpointing & resuming](#checkpointing--resuming)
13. [Verification suite](#verification-suite)

---

## Architecture

An OpenMythos forward pass is a three-stage flow:

| Stage | Executions | Role |
|---|---|---|
| **Prelude** (`prelude_layers`, default 2) | once | Projects token embeddings into a continuous latent state representation `e` via standard dense transformer blocks. |
| **Recurrent core** (`recurrent_layers` base blocks, `max_loop_iters` executions) | `T × R` | A stack of `recurrent_layers` weight-shared blocks executed `T = max_loop_iters` times, refining hidden state `h_t → h_{t+1}` conditioned on the frozen latent `e`. |
| **Coda** (`coda_layers`, default 2) | once | Dense post-processing on `h_T`, followed by RMSNorm and the vocabulary projection. |

Inside every recurrent step:

```text
g₁, g₂      = FeatureModulation(e)                    # latent gates (AdaLN-style)
u           = Attention(RMSNorm(h) · g₁)
h          += LoRA_attn[t](u)                         # depth-wise adapter (rank r=16)
m           = MoE_topK_shared(RMSNorm(h) · g₂)
h          += LoRA_moe[t](m)                          # second depth-wise adapter
h_{t+1}    = exp(dt·A_cont) ⊙ h  +  σ(gate(e)) · B e  # LTI injection (ρ(A)<1)
```

The heavy backbone (attention projections, expert FFNs) is **shared across all
T iterations**; only rank-16 adapters are indexed per `(block, t)` pair, so
inference-time compute scales linearly in T while parameters do not.

### Attention backbones

- **GQA** (`attn_type="gqa"`) — Grouped-Query Attention with RoPE, dispatched to
  FlashAttention-3 → FlashAttention-2 → torch SDPA depending on availability.
- **MLA** (`attn_type="mla"`) — DeepSeek-style Multi-Latent Attention: keys and
  values are compressed into a shared low-rank latent (`kv_lora_rank`, cached at
  inference), with a decoupled RoPE lane (`rope_head_dim`) bypassing compression.

### Sparse MoE feed-forward

DeepSeek-style routing with unconditional **shared experts** + routed experts,
top-k selection with renormalised softmax gates computed in fp32, drop-less
batched gather/scatter execution, and an auxiliary load-balancing loss
(`aux_loss_coeff`, Switch-style, evaluates to ≈1.0 under balanced routing).

## Theoretical background

**Why recurrent depth?** Scaling test-time computation is becoming a first-class
axis alongside parameter count. Weight-tied loops let a model refine its answer
inside a fixed-parameter budget; OpenMythos's recurrent core generalises the
"universal transformer" idea with two key upgrades that make deep unrolls
actually trainable:

1. **LTI spectral stability.** The linear injection uses a *continuous-time*
   negative diagonal matrix parameterised as `A_cont = -exp(log_a)` and
   discretised via Zero-Order Hold / Euler integration:

   ```text
   A_disc = exp(dt · A_cont),   dt = softplus(raw_dt) ≥ 1e-3
   ```

   Every diagonal entry of `A_disc` lies strictly inside `(0, 1)` — hence the
   spectral radius satisfies `ρ(A) < 1` **for every parameter value the
   optimiser can reach**, so activations cannot explode over long unrolls and
   gradients flowing backwards through the loop stay bounded. The logs are
   additionally clamped so the guarantee holds under fp32 rounding, not just
   symbolically.

2. **Depth-wise LoRA adapters.** Shared-weight looping collapses all iterations
   into one transformation; rank-16 adapters indexed by loop step `t`
   re-specialise each iteration at negligible cost. They are zero-initialised
   *and* tanh-gated at zero, making every loop step an exact identity at init:
   training starts at effective depth `P + C` and grows smoothly toward
   `P + T·R + C`.

**Test-time adaptivity:** pass `loop_iters=` at inference to trade compute for
quality without any retraining (train-short / test-deep).

## Repository layout

```text
openmythos/
├── .gitignore                  # Python/PyTorch/ckpt hygiene
├── README.md                   # this document
├── requirements.txt            # pip dependencies (torch, flash-attn, HF, ...)
├── setup.py                    # editable install of the openmythos package
├── train.py                    # CLI entrypoint (single-GPU or torchrun)
└── openmythos/
    ├── __init__.py             # lazy public surface
    ├── config.py               # MythosConfig + 100m…10b variant presets
    ├── attention.py            # GQAttention, MLAttention, RoPE, kernel dispatch
    ├── moe.py                  # SharedExpert, RoutedExpert, MythosMoE router
    ├── lti_recurrent.py        # LTIRecurrentInjection (ZOH), DepthWiseLoRA
    ├── model.py                # Prelude/Recurrent/Coda + OpenMythosForCausalLM
    ├── dataset.py              # streaming FineWeb-Edu token packing pipeline
    ├── gpu_profile.py          # 61-GPU DB, auto-detection, MFU tuning profiles
    ├── precision.py            # bf16/fp16/fp8/fp4 contexts + Blackwell detection
    └── utils.py                # param census, LR schedule, DDP/FSDP, loggers
```

The repo also ships standalone tooling under `scripts/`: `tune_gpu.py`
(GPU detection / MFU profiles / GEMM benchmark CLI) and
`sync_checkpoints_to_hf.py` (background checkpoint publisher).

## Installation

Requires **Python 3.11+** and CUDA-matched wheels for serious runs (CPU works
for bring-up and CI thanks to graceful kernel fallbacks).

```bash
git clone <your-fork-url> openmythos && cd openmythos

# 1. Torch matching your CUDA version, e.g. cu126/cu128/cu130 channel:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Remaining runtime deps:
pip install -r requirements.txt

# 3. The package itself (editable mode):
pip install -e .
```

Optional but recommended on Hopper/Blackwell nodes:

```bash
pip install flash-attn --no-build-isolation        # FA2 / FA3 fused kernels
pip install torchao>=0.5.0                          # torchao.float8 FP8 training
pip install "transformer_engine[pytorch]"           # TE FP8 recipes (H100/B200)
```

Every optional backend is probed at runtime; missing pieces degrade with a clear
one-line log message rather than an import error.

## Quickstart

Single GPU:

```bash
python train.py --variant 1b --precision bf16 --batch_size 8
```

Multi-GPU (8× GPUs, FP8 path, auto-selected FSDP sharding):

```bash
torchrun --nproc_per_node=8 train.py --variant 10b --precision fp8 \
    --grad_accum 4 --seq_len 4096 --max_steps 50000
```

Blackwell NVFP4 research run (auto-detected SM100/SM120 hardware):

```bash
python train.py --variant 1b --precision fp4 --batch_size 8
```

Minimal CPU bring-up (synthetic DEMO corpus when offline, warnings included):

```bash
HF_DATASETS_OFFLINE=1 python train.py --variant 100m --max_steps 20 \
    --seq_len 512 --batch_size 2 --tokenizer_name char256 --disable_wandb
```

Key CLI flags (see `--help` for the full set): `--variant`, `--dataset_name`,
`--batch_size`, `--grad_accum`, `--seq_len`, `--max_steps`, `--lr`,
`--precision {bf16,fp8,fp4,fp32}`, `--loop_iters`, `--use_flash_attn /
--no-use_flash_attn`, `--checkpoint_dir`, `--wandb_project`,
plus production extras (`--resume`, `--eval_interval`, `--grad_checkpoint`,
`--dist_strategy`, `--attn_type`, `--tokenizer_name`,
`--shuffle_buffer_docs`, `--low_ram`, `--aux_loss_coeff`,
`--z_loss_coeff`, …).

On RAM-constrained hosts (< 32 GB) add `--low_ram --num_workers 2 --batch_size 2`:
the data path stays strictly constant-RAM end to end — HF streaming pulls one
parquet row-group at a time, the shuffle reservoir is capped at 512 documents
and pinned-memory staging is disabled — so peak host memory is dominated by the
model/optimizer, not the corpus.

If your box still OOMs because parquet downloads outpace tokenisation, the
stream ships a built-in **download pacing gate**: `--tokenize_chunk_docs`
(default 30) pulls exactly N raw documents, fully tokenises + packs that batch,
then naps `--tokenize_pause_s` seconds (default 0.05) before resuming the
download — so buffered tokens stay bounded no matter how fast the link is. For
pathologically slow hosts use e.g.
`--low_ram --num_workers 1 --tokenize_pause_s 1.0`; progress and pack-buffer
depth are logged every 50 chunks so you can watch it pace itself.

## Model variants

Presets live in [`openmythos/config.py`](openmythos/config.py). Parameter counts
below were **measured on meta-device tensors from the actual module graph**
(includes embeddings, control planes, adapters and every routed expert) using
the default vocabulary 50257 and `max_loop_iters=8`:

| Variant | d_model | heads / kv | experts (routed + shared) | seq | Total params | Active params/token |
|---|---|---|---|---|---|---|
| `100m` | 640 | 10 / 2 | 12 + 1 | 2048 | **≈0.10 B** | ≈58 M |
| `300m` | 768 | 12 / 3 | 24 + 1 | 2048 | **≈0.31 B** | ≈99 M |
| `500m` | 1024 | 16 / 4 | 32 + 1 | 4096 | **≈0.49 B** | ≈136 M |
| `1b` | 1536 | 24 / 6 | 24 + 1 | 4096 | **≈0.91 B** | ≈340 M |
| `3b` | 2560 | 40 / 10 | 32 + 1 | 4096 | **≈2.89 B** | ≈765 M |
| `7b` | 3840 | 30 / 10 | 40 + 1 | 4096 | **≈7.59 B** | ≈1.54 B |
| `10b` | 4480 | 35 / 7 | 40 + 1 | 4096 | **≈10.23 B** | ≈1.99 B |

Design philosophy: because stacking additional *base* blocks multiplies the
expert pool, presets prefer **more loop iterations over more stacked blocks** —
raising `--loop_iters` at train time (or inference time) is far cheaper than
re-widening the stack, which is precisely the point of recurrent depth.

Any preset can be overridden per-field, e.g.
`MythosConfig.from_variant("1b", attn_type="mla", max_loop_iters=16)`.

## Precision scaling guide

Selected via `--precision`; managed by `openmythos/precision.py`.

| Mode | What runs where | Hardware requirement |
|---|---|---|
| `fp32` | Full float32; `torch.set_float32_matmul_precision('high')` enables TF32 GEMMs on Ampere+. | any |
| `bf16` | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`. Mixed-precision reduce stats configured automatically under FSDP/NCCL. | Ampere+ recommended |
| `fp16` | `torch.autocast(device_type='cuda', dtype=torch.float16)` with a **dynamic `GradScaler`** (auto-enabled by the trainer: losses are scaled before `backward`, gradients unscaled before clipping, `scaler.step/update` drive the optimizer). On Ampere+ a warning suggests `bf16` instead (same GEMM rate, wider mantissa). | Volta/Turing tensor cores |
| `fp8` | Backend ladder: **Transformer Engine** delayed scaling (HYBRID E4M3/E5M2 recipes, swapped-in `te.Linear`) → **torchao.float8** `convert_to_float8_training` + autocast context → clearly logged BF16 fallback. Sensitive ops (router, latent compressor, LM head) always remain high precision. | Hopper/Blackwell for speedup; safe anywhere |
| `fp4` | Blackwell-native **NVFP4 micro-scaled** quantisation path targeting SM100 (B200/GB200) / SM120 consumer dies: eligible linears become block-wise group-of-16 E2M1 micro-scaled ops. On non-Blackwell hardware the identical numeric contract is preserved through a portable straight-through-estimator emulation, with a one-line warning recommending `--precision fp8`. Control-plane projections are exempt. | Blackwell for native throughput |

Precision helpers honour the specification's contract exactly:

```python
from openmythos.precision import get_autocast_context, prepare_model_for_precision

model = prepare_model_for_precision(model, "fp4")      # before DDP/FSDP wrap
with get_autocast_context("fp4"):
    logits, loss, info = model(x, y)
    loss.backward()
```

`openmythos.utils.count_parameters` reports both total and *active-per-token*
parameters so routing efficiency is visible in logs from step zero.

## GPU auto-tune — MFU profiles for every card

`openmythos/gpu_profile.py` ships a curated database of **61 NVIDIA GPUs** —
Tesla P100/P40/P4, V100, T4, the GTX 16xx / RTX 20xx / 30xx / 40xx / 50xx
consumer stacks, A-series datacenter (A100/A800/A30/A10/A40/A2), L4/L20/L40/
L40S, RTX A/Ada workstations, H100/H200/H800/GH200/H20, and both Blackwell
generations (B100/B200/GB300-class and RTX 50xx / RTX PRO 6000). **Every
card has its own hand-researched tuning entry** (the `TUNE_DB`): a precision
ladder, the torch.compile mode that pays off on that die, SDPA kernel
priorities, per-architecture cuBLAS/NCCL/allocator env knobs, a data-feed
pacing plan that keeps the GPU fed, checkpointing bias, and a `kernel_eff`
figure — the fraction of the datasheet dense peak a well-tuned large GEMM
actually sustains on that silicon, with an architecture rationale for each.
Unknown cards fall back to a profile synthesised from their compute
capability, so nothing crashes.

The tuner's job is an **explicit MFU target (30% by default)**. A roofline
engine per card computes:

- the **ridge point** `peak TFLOPS / GB/s` (FLOP per byte moved) of the die,
- the **arithmetic intensity** of a training micro-step as a function of
  micro-batch, including weight traffic, AdamW/DDP amortisation, activation
  traffic and checkpoint recompute,
- a **wave factor** (does a d_model-wide GEMM even fill the SMs at this
  batch?),
- and it then **sweeps micro-batch and the checkpointing on/off option** to
  maximise projected MFU, preferring the smallest batch that already meets
  the target. The banner prints the projection, the gap, and an honest
  verdict: bandwidth-bound cards and pre-tensor-core silicon are told they
  cannot reach 30% and why — the target is never faked.

Two one-line integrations keep training honest on any silicon:

- **`--auto_tune`** detects the GPU, applies env tweaks *before* CUDA init,
  prints the full profile **with the MFU plan**, and fills every un-set
  tuning flag (precision, batch, accumulation, workers, flash attention,
  checkpointing, compile mode, data-feed pacing). Explicit flags always
  win; on hosts with no GPU it is a loud no-op.
- The **MFU gauge** now divides by the *detected card's* dense peak for the
  running precision instead of a hardcoded H100 constant — the `mfu/estimate`
  logged metric is truthful from a T4 to a B300.

```bash
# inspect this host
python scripts/tune_gpu.py                     # pretty report + MFU plan + launch cmd
python scripts/tune_gpu.py --target 0.35       # aim at a different MFU target
python scripts/tune_gpu.py --json              # machine-readable
python scripts/tune_gpu.py --env               # eval $(... --env) in shells
python scripts/tune_gpu.py --list              # dump the whole GPU DB

# what WOULD this run look like elsewhere?
python scripts/tune_gpu.py --simulate T4 --variant 100m
python scripts/tune_gpu.py --simulate B300 --variant 7b
python scripts/tune_gpu.py --simulate "cc=9.0,vram=80,sms=132"

# measure real GEMM throughput vs datasheet peak (needs a GPU)
python scripts/tune_gpu.py --bench

# validate the DB + every profile anywhere (CPU-only safe)
python scripts/tune_gpu.py --self-test         # 136/136 checks

# and just train with it
python train.py --auto_tune --variant 500m
HF_DATASETS_OFFLINE=0 torchrun --nproc_per_node=8 train.py --auto_tune --variant 10b
```

Representative tuned outcomes (the MFU plan is per card; `proj` = projected
MFU vs the 30% target):

| Card | Variant | Plan | proj | Notes |
|---|---|---|---|---|
| Tesla T4 ×2 (16 GB) | 100m s2048 | `fp16` 10×6, no-ckpt, compile `reduce-overhead` | 29.7% GAP | 320 GB/s + 70 W: kernel-eff ceiling; CUDA graphs are the lever |
| RTX 3090 (24 GB) | 500m s4096 | `bf16` 8×4, FA2, compile `reduce-overhead` | 32.2% MET | classic defaults, now justified per die |
| RTX 4090 (24 GB) | 500m s4096 | `bf16` 10×3, FA2, compile `max-autotune` | 26.7% GAP | 24 GB cannot fit a wave-filling batch; use ×2 GPUs |
| A100 80GB | 100m s2048 | `bf16` 17×4, FA2, no-ckpt, `max-autotune` | 30.9% MET | checkpointing off: recompute was burning FLOPs |
| H100 SXM (80 GB) | 100m s2048 | `bf16`/`fp8` 19×3, FA2, no-ckpt | 30.1% MET | fp8 needs TE/torchao installed |
| B300 (288 GB) | 100m s2048 | `bf16`/`fp8` 26×2, cuDNN SDPA, `max-autotune` | 30.3% MET | CUDA 12.8+, driver ≥ 570 |
| RTX 5050 (8 GB) | 500m s4096 | `bf16` 1×32, cuDNN SDPA | 9.8% GAP | honest ceiling: 224 GB/s entry die |

The trainer logs `mfu/estimate` every step — compare it against the banner's
projection; a large shortfall means the data feed is starving the GPU (raise
`--tokenize_chunk_docs`), not the silicon.

Security/robustness notes: multi-GPU GeForce rigs automatically get
`NCCL_P2P_DISABLE=1` (consumer cards lack reliable P2P), and every profile
exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to tame
fragmentation on small cards.

## Training stability — loss spikes, fp16 skips, router health

Short spikes on a MoE + recurrent loss curve (e.g. 5.2 → 6.9 for a step,
then recovery) have two very different origins, and the trainer now
instruments both.

**Router health (the big one).** Prior builds had a silent defect: with
`--grad_checkpoint` active, the MoE load-balancing loss was read from a
module attribute *after* the checkpointed (no-grad) forward pass, so it
was detached from the autograd graph — balancing never sent a gradient,
the router drifted, and the run showed exactly the
spike-then-recover signature. `RecurrentBlock.forward` now *returns* its
`(aux, z)` routing losses so they ride through the checkpoint recompute,
and a **router z-loss** (ST-MoE) is applied on top:

- `--z_loss_coeff` (default `1e-3`) penalises the squared log-partition
  of the router logits, damping the top-k assignment flip-flops that
  cause transient spikes; set `0` to disable.
- The balance loss is collected per loop *execution* (the weight-shared
  stack re-routes every iteration), not once per block.
- Logged keys `loss/aux`, `loss/z`, `moe/router_entropy`: healthy runs
  keep aux near 1.0 and entropy near `ln(num_experts)`.

**fp16 GradScaler churn.** Overflow steps (fp16 on T4/V100-class cards)
are now detected by scale comparison, counted, and no longer consume an
LR-schedule step:

- `--scaler_init_scale` (default 65536) and `--scaler_growth_interval`
  (default 2000; raise to 4000+ if skips repeat every few hundred steps);
- logged keys `train/grad_norm`, `train/loss_scale`,
  `train/skipped_steps`;
- an EMA-based warning prints grad-norm, router entropy, aux/z and the
  loss scale the moment a spike is detected, separating one-off hard
  batches (recover by themselves) from systematic scale churn (grad_norm
  pinned at the clip value with a rising skip count).

Regression coverage: `python scripts/test_routing_stability.py`
(12 deterministic CPU checks) proves the routing-loss gradients survive
gradient checkpointing on both attention backbones.

## Dataset acquisition & streaming

`get_fineweb_dataloader()` has two acquisition modes:

**`--data_mode native` (default) — HF-native download, timeout-proof.**
Parquet shards are pulled exactly once by `huggingface_hub` (resumable ranged
HTTP; 302/xet-bridge redirects handled internally; concurrency-safe file
locks), landing in the standard HF cache. Rows are then read from the *local*
files through pyarrow row-group batches. Because no live HTTP connection is
held while tokenisation runs, CDN read/idle timeouts during tokenizer pauses
are structurally impossible, and downloads survive process restarts.

- `--max_parquet_shards` (default 64) bounds disk usage (~100–300 MB per
  shard); cached shards are reused across runs and shared between ranks.
- **Any text-column corpus works**: the parquet schema is inspected and a
  well-known text column is picked automatically (`text`, `content`, `txt`,
  `body`, …), falling back to the first string column — e.g.
  `ProCreations/Ultra-FineWeb-EDU` (single `content` column) needs zero
  configuration.
- When the shard window is consumed, its order is deterministically reshuffled
  (endless training); raise the cap for more unique data. Rotation logs are
  throttled, and if more readers exist than shards (e.g. 2 GPUs × 2 workers
  over a single-shard dataset) the readers share shards via deterministic
  row-batch striding instead of starving.
- Works fully offline once shards are cached; `--low_ram`, the shuffle
  reservoir (`--shuffle_buffer_docs`) and the pacing gate
  (`--tokenize_chunk_docs/--tokenize_pause_s`) behave identically as before —
  with native mode they pace local reads instead of live sockets.
- Pre-mirrored corpus? Point `HF_MIRROR`-style environments or pass a local
  directory via `get_fineweb_dataloader(local_corpus_dir=...)`.

**`--data_mode stream` — legacy live reader.** The previous
`load_dataset(..., streaming=True)` path with buffer shuffling remains for
disk-constrained setups: on-the-fly tiktoken tokenisation, continuous packing,
automatic rank/worker sharding under `torchrun`, exponential-backoff reconnects
and DEMO fallback when nothing can be opened.

In both modes each window of `seq_len` tokens is self-shifted for causal LM
supervision, vocab size auto-propagates into the model config, and
`httpx` request-level INFO logging is silenced so your console shows training
progress rather than redirect handshakes.

Swap corpora freely: `--dataset_name HuggingFaceFW/fineweb` (or any text-column
corpus) works unchanged.

## Distributed training

- `openmythos.utils.setup_distributed()` initialises NCCL under `torchrun` and
  returns rank/world/device state; single-GPU runs skip entirely.
- `wrap_model(strategy='auto')` chooses **FSDP FULL_SHARD** (bf16 mixed
  precision, transformer-block auto-wrap policy, async all-gathers) for large
  variants and plain **DDP** for small ones; MoE-aware unused-parameter
  handling is applied automatically for DDP.
- Per-loop-step gradient checkpointing (`--grad_checkpoint`) trades recompute
  for VRAM when unrolling deep `T`.
- Loss is averaged consistently across grad accumulation × world size; grads
  clipped globally (1.0 default) before the fused AdamW step.

## Checkpointing & resuming

`save_checkpoint` writes atomically (`tmp → rename`) containing model/optimizer/
scheduler state, RNG state, config dict and CLI args at `-latest` intervals and
whenever validation improves (`-best`). Resume identically sized jobs with:

```bash
python train.py --variant 1b --resume checkpoints/1b-latest.pt
```

FSDP users should keep the same world size and strategy when resuming.

## Automatic checkpoint backup to HuggingFace Hub

`scripts/sync_checkpoints_to_hf.py` is a **torch-free, CPU-only daemon** that
runs alongside training (it never blocks or slows the trainer — checkpoints are
saved atomically by `train.py`, so the watcher can never observe partial
files), creates your model repo on first run, and pushes weights whenever they
actually change:

```bash
# one-time: hand the token to your shell (rotate it if it ever leaks!)
read -s HF_TOKEN && export HF_TOKEN

# launch in background, logs to sync.log
nohup python scripts/sync_checkpoints_to_hf.py \
    --checkpoint_dir ./checkpoints \
    --repo_id <your-user>/openmythos-500m \
    > sync.log 2>&1 &   echo $! > sync.pid
```

Behaviour and knobs:

- scans every `--poll_interval` seconds (default 300); a file is pushed only
  when its SHA-256 truly differs from the last pushed copy;
- per-file cooldown via `--min_upload_interval` (default 30 min) so rapid
  saves don't hammer the hub; dedup state survives restarts
  (`~/.cache/openmythos_sync/`);
- files land under `checkpoints/…` in the repo; every commit message embeds
  the sha256 prefix + UTC timestamp; full version history lives on the hub;
- repo is created private by default (`--public` to flip);
- test a single cycle with `--once` (cron-friendly);
- stop with `kill $(cat sync.pid)`.

Token precedence: `--hf_token` argument → `$HF_TOKEN` → `$HUGGING_FACE_HUB_TOKEN`
→ previously stored `huggingface-cli login`. The script never writes tokens to
disk. If a token ever appears in chat logs, shell history or screenshots,
revoke it at https://huggingface.co/settings/tokens immediately.

## Verification suite

Development smoke tests cover module imports, GQA/MLA forward-backward,
gradient-checkpointed loops, runtime loop-depth overrides, LTI spectral-radius
guarantees under adversarial parameter drift, LoRA identity-at-init, NVFP4
fake-quant error bounds (~0.09 relative RMS on gaussian tensors), LR-schedule
shape, MoE balance-loss calibration, offline DEMO streaming, and full per-variant
parameter census — see `scripts/smoke_test_openmythos.py` in the project
workspace for a reference harness you can paste into CI.
