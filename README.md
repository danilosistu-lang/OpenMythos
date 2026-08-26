OpenMythos — Recurrent-Depth Transformer (RDT)

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
7. [Precision scaling guide (BF16 / FP8 / NVFP4)](#precision-scaling-guide)
8. [Dataset streaming](#dataset-streaming)
9. [Distributed training](#distributed-training)
10. [Checkpointing & resuming](#checkpointing--resuming)
11. [Verification suite](#verification-suite)

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
    ├── precision.py            # bf16/fp8/fp4 contexts + Blackwell detection
    └── utils.py                # param census, LR schedule, DDP/FSDP, loggers
```

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
`--dist_strategy`, `--attn_type`, `--tokenizer_name`, …).

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

## Dataset streaming

`get_fineweb_dataloader()` streams **HuggingFace FineWeb-Edu** with no local
cache requirement:

- `datasets.load_dataset(..., streaming=True)` plus buffer shuffling;
- on-the-fly tokenisation via tiktoken (`gpt2` byte-level BPE by default;
  `cl100k_base` also supported — vocab size auto-propagates into the config);
- continuous concatenation packing into fixed `seq_len` windows (each window is
  self-shifted for causal LM supervision);
- automatic rank & worker sharding under `torchrun`;
- transient network failures trigger exponential-backoff reconnects mid-stream,
  while unresolvable cold-starts degrade into a loudly-labelled synthetic DEMO
  corpus so pipelines never die on day zero;
- epoch rotation reshuffles deterministically forever (the loader is endless).

Swap corpora freely: `--dataset_name HuggingFaceFW/fineweb` (or any text column
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

## Verification suite

Development smoke tests cover module imports, GQA/MLA forward-backward,
gradient-checkpointed loops, runtime loop-depth overrides, LTI spectral-radius
guarantees under adversarial parameter drift, LoRA identity-at-init, NVFP4
fake-quant error bounds (~0.09 relative RMS on gaussian tensors), LR-schedule
shape, MoE balance-loss calibration, offline DEMO streaming, and full per-variant
parameter census — see `scripts/smoke_test_openmythos.py` in the project
workspace for a reference harness you can paste into CI.
