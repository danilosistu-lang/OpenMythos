"""Streaming dataset pipeline for OpenMythos.

Streams the Hugging Face ``FineWeb`` / ``FineWeb-Edu`` corpora token-by-token
with zero local caching, packs the document stream into fixed-length
token buffers suitable for causal language modelling, and partitions shards
across distributed ranks and DataLoader workers automatically.

Design highlights
-----------------
* ``streaming=True`` for infinite seamless reads (no full-corpus download).
* On-the-fly tokenisation with ``tiktoken`` (GPT-2 ``50257`` BPE by default,
  optional ``cl100k_base``).
* Continuous concatenation packing: document boundaries are discarded and a
  rolling buffer yields exactly ``seq_len``-sized windows, so every token is
  used as both context and target.
* Rank/worker sharding via ``datasets.IterableDataset.shard``, deterministic
  per (seed, epoch, rank, worker).
* Network-resilient: transient connection errors trigger exponential-backoff
  stream re-establishment instead of killing a multi-day training run.
* Graceful offline fallback: when neither HF datasets nor network access are
  available the loader synthesises a Zipf-distributed pseudo corpus and logs
  DEMO-mode banners, keeping the repository runnable out-of-the-box anywhere.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.utils.data as torch_data

logger = logging.getLogger("openmythos.dataset")

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"


# ===========================================================================
# Tokenisation
# ===========================================================================
class CharTokenizer:
    """Dependency-free byte-pair fallback tokenizer (offline safety net).

    Maps printable ASCII bytes onto ids ``[0, 255]``.  It exists solely so
    that air-gapped machines can still execute an end-to-end training run;
    use ``gpt2`` (default) or ``cl100k_base`` for serious work.
    """

    name = "char256"
    vocab_size = 257  # 256 byte lanes + one pad lane

    def encode(self, text: str) -> list:
        return [min(b, 256) for b in text.encode("utf-8", errors="replace")]


def build_tokenizer(name: str):
    """Return ``(tokenizer, vocab_size)`` honouring requested encoding."""
    if name == "char256":
        tok = CharTokenizer()
        return tok, tok.vocab_size
    try:
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return enc, enc.n_vocab
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "tiktoken tokenizer '%s' unavailable (%s); "
            "falling back to char256 offline tokenizer.",
            name,
            exc,
        )
        tok = CharTokenizer()
        return tok, tok.vocab_size


# ===========================================================================
# Core streaming iterable
# ===========================================================================
@dataclass
class StreamMeta:
    """Bookkeeping surfaced by :class:`FineWebStreamDataset`."""

    dataset_name: str
    split: str
    tokenizer_name: str
    vocab_size: int
    seq_len: int
    demo_mode: bool = False
    epochs_rotated: int = 0


def _sync_world() -> Tuple[int, int]:
    """Best-effort (rank, world_size) from torch.distributed state."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    env_rank = int(os.environ.get("RANK", 0))
    env_world = int(os.environ.get("WORLD_SIZE", 1))
    return env_rank, max(env_world, 1)


_HF_LOADABLE_CACHE: Optional[bool] = None


def _hf_datasets_importable() -> bool:
    """One-time capability probe so offline boxes never spin in retry loops."""
    global _HF_LOADABLE_CACHE
    if _HF_LOADABLE_CACHE is None:
        try:
            import datasets  # noqa: F401

            _HF_LOADABLE_CACHE = True
        except Exception:
            _HF_LOADABLE_CACHE = False
    return _HF_LOADABLE_CACHE


class FineWebStreamDataset(torch_data.IterableDataset):
    """Endless packed-token stream over a HF text corpus."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        seq_len: int = 4096,
        seed: int = 1337,
        shuffle_buffer_docs: int = 8192,
        streaming_shuffle: bool = True,
        demo_mode_fallback: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer_docs = shuffle_buffer_docs
        self.streaming_shuffle = streaming_shuffle
        self.demo_mode_fallback = demo_mode_fallback

        self.tokenizer, self.vocab_size = build_tokenizer(tokenizer_name)
        if os.environ.get("HF_DATASETS_OFFLINE", "0") == "1" \
                or not _hf_datasets_importable():
            self.demo_mode_fallback = True
            self._force_demo = True
        else:
            self._force_demo = False
        self.meta = StreamMeta(
            dataset_name=dataset_name,
            split=split,
            tokenizer_name=tokenizer_name,
            vocab_size=self.vocab_size,
            seq_len=seq_len,
            demo_mode=False,
        )
        self._warned_demo_banner = False

    # ------------------------------------------------------------ plumbing
    def _worker_id(self) -> Tuple[int, int]:
        info = torch_data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def _open_hf_stream(self, epoch: int):
        from datasets import load_dataset  # deferred: heavy import

        rank, world = _sync_world()
        worker_id, n_workers = self._worker_id()

        shard_index = rank * n_workers + worker_id
        total_shards = world * n_workers

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.streaming_shuffle:
            ds = ds.shuffle(
                seed=self.seed + epoch * 977, buffer_size=self.shuffle_buffer_docs
            )
        ds = ds.shard(num_shards=total_shards, index=shard_index, contiguous=True)
        return iter(ds)

    def _demo_stream(self, epoch: int):
        """Deterministic synthetic Zipf-ish corpus for offline demo runs."""
        rng = random.Random(self.seed * 31 + epoch)
        vocab_span = max(64, min(self.vocab_size - 1, 32768))
        weights = [1.0 / (i + 8) ** 1.15 for i in range(vocab_span)]
        while True:                       # docs of plausible length
            length = rng.randint(96, 768)
            yield {"text_tokens": [rng.choices(range(vocab_span), weights)[0]
                                   for _ in range(length)]}

    def _raw_doc_iter(self, epoch: int) -> Iterator[Dict[str, Any]]:
        """Yield raw documents; degrades cleanly to the synthetic corpus.

        Failure policy:
        * ``datasets`` package absent / forced-offline   -> immediate DEMO mode.
        * initial open failures after ``max_open_tries``  -> DEMO mode.
        * mid-stream network errors                       -> exponential-backoff
          reconnection, keeping long training runs alive across node flaps.
        """
        if getattr(self, "_force_demo", False):
            logger.warning(
                "[dataset] 'datasets' unavailable or offline-mode requested - "
                "entering DEMO mode with synthetic data. Results are NOT meaningful!"
            )
            self.meta.demo_mode = True
            yield from self._demo_stream(epoch)
            return

        max_open_tries = 6
        open_failures = 0
        ever_opened = False
        backoff = 2.0

        while True:
            try:
                inner = self._open_hf_stream(epoch)
                ever_opened = True
                open_failures = 0
                while True:
                    yield next(inner)
            except StopIteration:
                # Split exhausted this pass: rotate the shuffle seed so the
                # endless stream keeps producing fresh orderings per epoch.
                logger.info(
                    "[dataset] split '%s' exhausted; rotating shuffle seed "
                    "(epoch %d done).", self.split,
                    self.meta.epochs_rotated,
                )
                self.meta.epochs_rotated += 1
                epoch = self.meta.epochs_rotated
                continue
            except KeyboardInterrupt:
                raise
            except Exception as exc:                      # noqa: BLE001
                if not ever_opened:
                    open_failures += 1
                    if not self.demo_mode_fallback:
                        raise RuntimeError(
                            f"cannot open '{self.dataset_name}' ({exc}) and "
                            "demo_mode_fallback=False"
                        ) from exc
                    if open_failures >= max_open_tries:
                        logger.warning(
                            "[dataset] failed to open '%s' %d times (%s). "
                            "Entering DEMO mode with synthetic offline data - "
                            "results are NOT meaningful!",
                            self.dataset_name, open_failures, exc,
                        )
                        self.meta.demo_mode = True
                        yield from self._demo_stream(epoch)
                        return
                    sleep_s = min(backoff * (2 ** min(open_failures, 4)), 60.0)
                    logger.warning(
                        "[dataset] open attempt %d/%d failed (%s); "
                        "retrying in %.1fs",
                        open_failures, max_open_tries, exc, sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    # Mid-stream drop: standard production behaviour is to
                    # reconnect indefinitely (corpora >> any outage window).
                    sleep_s = min(backoff * (2 ** (open_failures)), 300.0)
                    logger.warning(
                        "[dataset] stream interrupted (%s); reconnecting in %.1fs",
                        exc, sleep_s,
                    )
                    time.sleep(sleep_s)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.meta.demo_mode and not self._warned_demo_banner:
            self._warned_demo_banner = True

        buffer: list = []
        start = 0                                   # rolling window pointer
        emit_len = self.seq_len
        epoch = self.meta.epochs_rotated
        for doc in self._raw_doc_iter(epoch):
            if isinstance(doc, dict) and "text_tokens" in doc:
                tokens = doc["text_tokens"]
            else:
                text = doc.get("text", "")
                if not text:
                    continue
                tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            while len(buffer) - start >= emit_len + 1:
                window = buffer[start : start + emit_len + 1]   # inputs + shifted label
                x = torch.tensor(window[:-1], dtype=torch.long)
                y = torch.tensor(window[1:], dtype=torch.long)
                start += emit_len
                if start >= (1 << 20):                       # amortised compaction
                    del buffer[:start]
                    start = 0
                yield x, y


def _network_reachable(host: str = "huggingface.co", timeout: float = 2.5) -> bool:
    """Cheap TCP reachability probe used to shortcut hopeless retries."""
    import socket

    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except OSError:
        return False


# ===========================================================================
# Public factory (spec-mandated signature)
# ===========================================================================
def get_fineweb_dataloader(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    batch_size: int = 8,
    seq_len: int = 4096,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2",
    seed: int = 1337,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> Tuple[torch_data.DataLoader, StreamMeta]:
    """Build an endless DataLoader over FineWeb(-Edu).

    Returns:
        ``(dataloader, meta)`` where ``meta.vocab_size`` must be consumed by
        the caller *before* model construction so embeddings match the active
        tokenizer.
    """
    dataset = FineWebStreamDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        seed=seed,
    )
    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=max(prefetch_factor, 2) if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
    )
    return loader, dataset.meta


__all__ = [
    "FineWebStreamDataset",
    "StreamMeta",
    "CharTokenizer",
    "build_tokenizer",
    "get_fineweb_dataloader",
]
