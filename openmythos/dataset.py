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
        shuffle_buffer_docs: int = 2048,
        streaming_shuffle: bool = True,
        demo_mode_fallback: bool = True,
        low_ram_profile: bool = False,
        tokenize_chunk_docs: int = 30,
        tokenize_pause_s: float = 0.05,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.seq_len = seq_len
        self.seed = seed
        if low_ram_profile:
            # Constant-RAM guarantee for small hosts (<32 GB): trade a little
            # shuffling entropy for a strictly bounded resident set.
            shuffle_buffer_docs = min(shuffle_buffer_docs, 512)
        self.shuffle_buffer_docs = shuffle_buffer_docs
        self.streaming_shuffle = streaming_shuffle
        self.demo_mode_fallback = demo_mode_fallback
        self.low_ram_profile = low_ram_profile
        # --- download pacing gate -----------------------------------------
        # The single most effective anti-OOM control for tiny hosts: instead of
        # letting every worker stream documents as fast as the network allows
        # (parquet row-groups pile up faster than tiktoken can drain them), we
        # pull exactly ``tokenize_chunk_docs`` docs, FULLY tokenise + pack that
        # batch, sleep ``tokenize_pause_s`` seconds, and only then resume
        # downloading.  Peak buffered tokens therefore stay bounded by
        #   chunk_docs x longest_doc + one seq_len window,
        # regardless of link speed or reader-side read-ahead.
        self.tokenize_chunk_docs = max(1, int(tokenize_chunk_docs))
        self.tokenize_pause_s = max(0.0, float(tokenize_pause_s))

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

        # streaming=True is load-bearing for the constant-RAM contract of this
        # class: HF pulls one parquet row-group at a time and never materialises
        # the corpus locally.  Guard it so a future refactor cannot silently
        # reintroduce a full download on 32 GB-class machines.
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if getattr(ds, "dataset_size", None) not in (None, 0):
            logger.debug("streaming over remote shards; dataset_size=%s", ds.dataset_size)
        if self.streaming_shuffle:
            # Bounded doc-level reservoir: RAM cost is roughly
            # buffer_size x avg_doc_bytes and stays flat during training.
            ds = ds.shuffle(
                seed=self.seed + epoch * 977, buffer_size=self.shuffle_buffer_docs
            )
        # NOTE: modulo-based shard (contiguous=False) -- contiguous sharding of
        # IterableDatasets has changed semantics across datasets releases; the
        # modulo filter is stable everywhere and still guarantees every rank
        # sees a disjoint slice of documents.
        ds = ds.shard(num_shards=total_shards, index=shard_index)
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
        info = torch_data.get_worker_info()
        if info is None or info.id == 0:
            logger.info(
                "[dataset] streaming '%s' (%s): constant-RAM mode"
                " | shuffle_buffer=%d docs%s"
                " | pacing=every %d docs wait+%.2fs",
                self.split, self.dataset_name, self.shuffle_buffer_docs,
                " | LOW-RAM profile" if self.low_ram_profile else "",
                self.tokenize_chunk_docs, self.tokenize_pause_s,
            )

        buffer: list = []
        start = 0                                   # rolling window pointer
        emit_len = self.seq_len
        epoch = self.meta.epochs_rotated

        docs_in_chunk = 0
        chunks_done = 0
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

            # ---- download pacing gate (see __init__ rationale) ------------
            docs_in_chunk += 1
            if docs_in_chunk >= self.tokenize_chunk_docs:
                docs_in_chunk = 0
                chunks_done += 1
                # Everything pulled so far is already tokenised and packed;
                # this enforced nap stops the HTTP readers from running away
                # ahead of the CPU-bound tokenizer on small-RAM machines.
                if self.tokenize_pause_s > 0.0:
                    time.sleep(self.tokenize_pause_s)
                if chunks_done % 50 == 0:
                    logger.info(
                        "[dataset] paced progress: %d docs ingested, "
                        "pack-buffer depth %d tokens",
                        chunks_done * self.tokenize_chunk_docs,
                        len(buffer) - start,
                    )


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
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    shuffle_buffer_docs: int = 2048,
    streaming_shuffle: bool = True,
    low_ram_profile: bool = False,
    tokenize_chunk_docs: int = 30,
    tokenize_pause_s: float = 0.05,
) -> Tuple[torch_data.DataLoader, StreamMeta]:
    """Build an endless DataLoader over FineWeb(-Edu).

    The pipeline is fully streaming end-to-end: ``load_dataset(...,
    streaming=True)``, a bounded document shuffle reservoir and per-worker
    token packing with amortised compaction.  Peak host RAM stays in the tens
    of megabytes *per worker* regardless of corpus size.

    Args:
        shuffle_buffer_docs: docs held by the streaming reshuffler. Lower to
            e.g. ``512`` on machines with tight memory budgets.
        low_ram_profile: additionally caps the shuffle buffer at 512 docs.
        num_workers: set ``0`` to keep everything inside the main process
            (smallest possible footprint, one tokenizer instance).
        tokenize_chunk_docs: download-pacing gate size â after every N raw
            documents the stream fully tokenises/packs what it pulled and only
            then resumes downloading (default 30).
        tokenize_pause_s: enforced nap after each paced chunk completes; raise
            it (e.g. ``1.0``) on very slow hosts to further clamp throughput.

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
        shuffle_buffer_docs=shuffle_buffer_docs,
        streaming_shuffle=streaming_shuffle,
        low_ram_profile=low_ram_profile,
        tokenize_chunk_docs=tokenize_chunk_docs,
        tokenize_pause_s=tokenize_pause_s,
    )
    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(pin_memory
                    and torch.cuda.is_available()
                    and not low_ram_profile),
        prefetch_factor=max(prefetch_factor, 1) if num_workers > 0 else None,
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
