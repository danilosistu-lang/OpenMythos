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
# HF-native acquisition helpers (default data path)
# ===========================================================================
# Rationale for native-first downloads:
#   * ``huggingface_hub`` pulls shards ONCE into the local HF cache using
#     resumable ranged HTTP (302/xet redirects handled internally, verified
#     completion via atomic rename).  No live connection is held afterwards,
#     so tokenizer-side pauses can never trip CDN read/idle timeouts.
#   * Rows are then read from the LOCAL parquet files by pyarrow in bounded
#     batches -- constant RAM, zero per-request log noise, restartable.
NATIVE_STACK_OK: Optional[bool] = None


def _probe_native_stack() -> bool:
    """One-shot capability probe for the native hub download stack."""
    global NATIVE_STACK_OK
    if NATIVE_STACK_OK is None:
        try:
            import huggingface_hub  # noqa: F401
            import pyarrow  # noqa: F401

            NATIVE_STACK_OK = True
        except Exception:
            NATIVE_STACK_OK = False
    return NATIVE_STACK_OK


_REPO_PLAN_CACHE: Dict[str, list] = {}


def _resolve_native_plan(dataset_name: str, split: str,
                         limit: Optional[int]) -> list:
    """Deterministic, cache-backed list of remote ``*.parquet`` shards.

    Uses the official ``huggingface_hub.list_repo_files`` API so auth,
    pagination and redirect topology stay entirely HF-native.
    """
    key = f"{dataset_name}|{split}|{limit}"
    if key in _REPO_PLAN_CACHE:
        return _REPO_PLAN_CACHE[key]
    from huggingface_hub import list_repo_files

    every = list_repo_files(dataset_name, repo_type="dataset")
    parquet = sorted(f for f in every if f.lower().endswith(".parquet"))
    wanted_splits = {s.strip().lower() for s in split.split(",")}
    preferred = [
        f for f in parquet
        if any(s in f.lower().rsplit("/", 1)[0].split("/")
               for s in wanted_splits)
    ]
    plan = preferred or parquet       # e.g. FineWeb dump paths omit 'train'
    if limit:
        plan = plan[: int(limit)]
    if not plan:
        raise RuntimeError(f"no parquet shards listed for '{dataset_name}'")
    _REPO_PLAN_CACHE[key] = plan
    return plan


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
        data_mode: str = "native",
        max_parquet_shards: Optional[int] = 64,
        local_corpus_dir: Optional[str] = None,
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
        # --- acquisition mode ----------------------------------------------
        # "native": huggingface_hub snapshot download -> local pyarrow reads.
        # "stream": legacy live load_dataset(streaming=True) HTTP reader.
        self.data_mode = data_mode
        self.max_parquet_shards = max_parquet_shards
        self.local_corpus_dir = local_corpus_dir
        if self.data_mode == "native":
            if _probe_native_stack():
                self.data_mode = "native"
            else:
                logger.warning(
                    "[dataset] native stack unavailable (huggingface_hub / "
                    "pyarrow missing); falling back to live HF streaming."
                )
                self.data_mode = "stream"
        if self.data_mode == "stream" and (
            os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
            or not _hf_datasets_importable()
        ):
            self.demo_mode_fallback = True
            self._force_demo = True
        else:
            self._force_demo = False

        self.tokenizer, self.vocab_size = build_tokenizer(tokenizer_name)
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

    def _native_doc_iter(self, epoch: int) -> Iterator[Dict[str, Any]]:
        """Yield raw documents from HF-native, locally-cached parquet shards.

        Each shard is fetched exactly once via ``hf_hub_download`` (resumable,
        checksum-verified, concurrency-safe file locks); reading then happens
        entirely from disk row-group by row-group through pyarrow.  Because no
        live socket is held while we tokenise/pack, the download stops-and-
        resumes of the previous live-reader design cannot time out anymore.
        Disk cost is bounded by ``max_parquet_shards`` shards cached under
        ``$HF_HOME/hub`` (reused across runs and safely shareable).
        """
        import pyarrow.parquet as pq

        if self.local_corpus_dir:
            # Pre-mirrored corpus on local disk (also our offline test path).
            owned: list = [
                os.path.join(self.local_corpus_dir, name)
                for name in sorted(os.listdir(self.local_corpus_dir))
                if name.endswith(".parquet")
            ]
            local_for = lambda rel: rel                      # noqa: E731
        else:
            from huggingface_hub import hf_hub_download

            plan = _resolve_native_plan(
                self.dataset_name, self.split, self.max_parquet_shards
            )
            rank, world = _sync_world()
            worker_id, n_workers = self._worker_id()
            owner = rank * n_workers + worker_id
            owners = max(world * n_workers, 1)
            owned = [f for i, f in enumerate(plan) if i % owners == owner]
            local_for = lambda rel: hf_hub_download(         # noqa: E731
                repo_id=self.dataset_name, repo_type="dataset", filename=rel
            )

        rng = random.Random(self.seed * 4241 + epoch)
        rng.shuffle(owned)

        for rel in owned:
            local_path = local_for(rel)
            with pq.ParquetFile(local_path) as pf:
                for batch in pf.iter_batches(batch_size=512, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        if isinstance(text, str) and text:
                            yield {"text": text}

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
        """Yield raw documents from the configured acquisition source.

        Failure policy per mode:
        * ``native``  -- listing/download errors use the shared retry ladder;
          on-listed-plan exhaustion (finite shard window) the epoch rotates so
          training never stalls once the window has been consumed.
        * ``stream``  -- legacy live reader: cold-open failures retry with
          exponential backoff into DEMO mode; mid-stream drops reconnect.
        * forced-offline / missing libraries -> synthetic DEMO corpus.
        """
        if getattr(self, "_force_demo", False):
            logger.warning(
                "[dataset] offline-mode requested or libraries missing - "
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
                if self.data_mode == "native":
                    ever_opened = True
                    open_failures = 0
                    yield from self._native_doc_iter(epoch)
                    # Finite local plan exhausted -> rotate for endlessness.
                    logger.info(
                        "[dataset][native] shard window consumed; rotating "
                        "order (epoch %d done). Raise --max_parquet_shards "
                        "for more unique data.", self.meta.epochs_rotated,
                    )
                    self.meta.epochs_rotated += 1
                    epoch = self.meta.epochs_rotated
                    continue

                ever_opened = True
                inner = self._open_hf_stream(epoch)
                open_failures = 0
                while True:
                    yield next(inner)
            except StopIteration:
                # Legacy live split exhausted this pass.
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
                    # Established-then-dropped (listing ok but a later shard
                    # fetch died): production behaviour is indefinite backoff
                    # reconnect -- hf_hub_download resumes partial files.
                    sleep_s = min(backoff * (2 ** (open_failures)), 300.0)
                    logger.warning(
                        "[dataset] acquisition interrupted (%s); resuming "
                        "in %.1fs", exc, sleep_s,
                    )
                    time.sleep(sleep_s)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = torch_data.get_worker_info()
        if info is None or info.id == 0:
            logger.info(
                "[dataset] acquiring '%s' (%s) via data_mode=%s"
                " | shuffle_buffer=%d docs%s"
                " | pacing=every %d docs wait+%.2fs",
                self.split, self.dataset_name, self.data_mode,
                self.shuffle_buffer_docs,
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
# Pre-tokenised memmap corpus (offline tokenization path)
# ===========================================================================
# Why this exists
# ---------------
# Some training hosts (small laptops, e.g. 8 GB-class RTX 5050 notebooks)
# should not spend any RAM/CPU on tokenization at all.  A big-RAM machine can
# run ``scripts/pretokenize_corpus.py`` once to produce flat ``*.bin`` token
# files plus a ``meta.json``; those files are copied (USB disk / network
# share / scp) to the training host and read back via OS memory mapping.
# RAM cost on the reader side is effectively zero regardless of corpus size,
# because pages are paged in on demand and never copied into the heap.
_MEMMAP_VALID_DTYPES = ("uint16", "uint32")


class MemmapTokensDataset(torch_data.IterableDataset):
    """Endless packed-token windows over a pre-tokenised ``*.bin`` corpus.

    Expected directory layout (produced by ``scripts/pretokenize_corpus.py``)::

        tokens_dir/
            meta.json          # {"vocab_size", "dtype", "tokenizer", "files"...}
            train-00000.bin    # raw token-id array, flat little-endian uint16/uint32
            train-00001.bin
            ...
            val-00000.bin      # optional validation split

    The corpus is sliced into ``segment_tokens``-sized blocks; blocks are
    shuffled per epoch and partitioned modulo ``world_size * num_workers`` so
    every rank/worker owns a disjoint slice of windows.
    """

    def __init__(
        self,
        tokens_dir: str,
        split: str = "train",
        seq_len: int = 2048,
        seed: int = 1337,
        segment_tokens: int = 1 << 21,   # ~2M tokens/segment (~4 MB at uint16)
    ):
        super().__init__()
        import json

        self.tokens_dir = os.path.abspath(os.path.expanduser(tokens_dir))
        meta_path = os.path.join(self.tokens_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"no meta.json in '{self.tokens_dir}' - run "
                "scripts/pretokenize_corpus.py to build a pre-tokenised corpus"
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        dtype = str(manifest.get("dtype", "uint16"))
        if dtype not in _MEMMAP_VALID_DTYPES:
            raise ValueError(
                f"unsupported token dtype '{dtype}' in {meta_path} "
                f"(expected one of {_MEMMAP_VALID_DTYPES})"
            )
        self.dtype = dtype
        self.vocab_size = int(manifest["vocab_size"])
        self.tokenizer_name = str(manifest.get("tokenizer", "unknown"))
        self.seq_len = int(seq_len)
        self.seed = int(seed)

        wanted_prefix = "val" if split.strip().lower().startswith("val") else "train"
        files = sorted(
            f for f in os.listdir(self.tokens_dir)
            if f.endswith(".bin") and f.split("-", 1)[0] == wanted_prefix
        )
        if not files:
            if wanted_prefix == "val":
                # Validation is optional: silently reuse the training split.
                files = sorted(
                    f for f in os.listdir(self.tokens_dir)
                    if f.endswith(".bin") and f.startswith("train")
                )
            if not files:
                raise FileNotFoundError(
                    f"no '{wanted_prefix}*.bin' shards in '{self.tokens_dir}'"
                )
        self.files = [os.path.join(self.tokens_dir, f) for f in files]
        self.split = "val" if wanted_prefix == "val" else "train"
        self.segment_tokens = max(int(segment_tokens), self.seq_len + 1)

        self.meta = StreamMeta(
            dataset_name=os.path.basename(self.tokens_dir),
            split=self.split,
            tokenizer_name=self.tokenizer_name,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            demo_mode=False,
        )

    def _worker_id(self) -> Tuple[int, int]:
        info = torch_data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    # --------------------------------------------------------------- segments
    def _segments(self, epoch: int):
        """Yield ``(file_index, start, length)`` blocks for one epoch.

        A window never straddles a block boundary, so short tail fragments of
        each shard are discarded (at most ``segment_tokens`` tokens per shard,
        i.e. a negligible fraction of any real corpus).
        """
        import numpy as np

        np_dtype = np.dtype(self.dtype)
        itemsize = int(np_dtype.itemsize)
        need = self.seq_len + 1
        blocks: list = []
        for fi, path in enumerate(self.files):
            file_tokens = os.path.getsize(path) // itemsize
            if file_tokens < need:
                continue                    # shard cannot yield a single window
            if file_tokens >= self.segment_tokens:
                n_seg = file_tokens // self.segment_tokens
                for si in range(n_seg):
                    blocks.append(
                        (fi, si * self.segment_tokens, self.segment_tokens)
                    )
            else:
                # Small shard (or tiny test corpus): use the whole file as
                # one segment; the window loop discards only the short tail.
                blocks.append((fi, 0, file_tokens))
        if not blocks:
            raise RuntimeError(
                f"pre-tokenised corpus in '{self.tokens_dir}' contains no "
                f"window of {self.seq_len + 1} tokens for split '{self.split}'"
            )
        rng = random.Random(self.seed * 7919 + epoch)
        rng.shuffle(blocks)
        return blocks

    # ----------------------------------------------------------------- stream
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import numpy as np

        rank, world = _sync_world()
        worker_id, n_workers = self._worker_id()
        owner = rank * n_workers + worker_id
        owners = max(world * n_workers, 1)

        epoch = 0
        np_dtype = np.dtype(self.dtype)
        while True:
            blocks = self._segments(epoch)
            blocks = blocks[owner::owners]      # disjoint slice per rank/worker
            open_maps: dict = {}
            try:
                for fi, start, length in blocks:
                    if fi not in open_maps:
                        open_maps[fi] = np.memmap(
                            self.files[fi], dtype=np_dtype, mode="r"
                        )
                    tokens = np.asarray(open_maps[fi][start : start + length])
                    # Walk the contiguous segment in seq_len windows; each
                    # window is self-shifted for causal LM supervision.
                    for off in range(0, length - self.seq_len, self.seq_len):
                        window = tokens[off : off + self.seq_len + 1]
                        x = torch.from_numpy(window[:-1].astype(np.int64, copy=True))
                        y = torch.from_numpy(window[1:].astype(np.int64, copy=True))
                        yield x, y
            finally:
                open_maps.clear()      # drop memmap handles each epoch
            epoch += 1


def get_memmap_dataloader(
    tokens_dir: str,
    split: str = "train",
    batch_size: int = 8,
    seq_len: int = 2048,
    num_workers: int = 4,
    seed: int = 1337,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    segment_tokens: int = 1 << 21,
) -> Tuple[torch_data.DataLoader, StreamMeta]:
    """Build a DataLoader over a pre-tokenised memmap corpus.

    Near-zero host RAM: token ids are paged directly from disk by the OS;
    no tokenizer, no parquet buffers, no HF download ever runs on this
    machine.  Create the corpus on a big-RAM box with
    ``scripts/pretokenize_corpus.py``, copy the folder over, and point
    ``--local_tokens_dir`` at it.
    """
    dataset = MemmapTokensDataset(
        tokens_dir=tokens_dir,
        split=split,
        seq_len=seq_len,
        seed=seed,
        segment_tokens=segment_tokens,
    )
    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(pin_memory and torch.cuda.is_available()),
        prefetch_factor=max(prefetch_factor, 1) if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
    )
    return loader, dataset.meta


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
    data_mode: str = "native",
    max_parquet_shards: Optional[int] = 64,
    local_corpus_dir: Optional[str] = None,
) -> Tuple[torch_data.DataLoader, StreamMeta]:
    """Build an endless DataLoader over FineWeb(-Edu).

    The pipeline keeps host RAM constant end-to-end: HF-native shard
    downloads land in the local hub cache, a bounded document shuffle
    reservoir feeds per-worker tokenisation, and packing compacts its token
    window periodically.  Peak RAM stays in the tens of megabytes *per
    worker* regardless of corpus size.

    Args:
        shuffle_buffer_docs: docs held by the streaming reshuffler. Lower to
            e.g. ``512`` on machines with tight memory budgets.
        low_ram_profile: additionally caps the shuffle buffer at 512 docs.
        num_workers: set ``0`` to keep everything inside the main process
            (smallest possible footprint, one tokenizer instance).
        tokenize_chunk_docs: download-pacing gate size â after every N raw
            documents the stream fully tokenises/packs what it pulled and only
            then resumes consuming the source (default 30).
        tokenize_pause_s: enforced nap after each paced chunk completes.
        data_mode: ``"native"`` (default) downloads parquet shards once via
            huggingface_hub into the shared HF cache and then reads them
            locally through pyarrow -- no live HTTP held during tokenisation;
            ``"stream"`` keeps the legacy live load_dataset(streaming=True)
            reader.
        max_parquet_shards: disk-bounding cap on how many remote shards are
            pulled per corpus (~100-300 MB each); cached across runs.
        local_corpus_dir: optional pre-mirrored directory of *.parquet files;
            when set, overrides hub resolution entirely (offline corpora).

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
        data_mode=data_mode,
        max_parquet_shards=max_parquet_shards,
        local_corpus_dir=local_corpus_dir,
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
    "MemmapTokensDataset",
    "StreamMeta",
    "CharTokenizer",
    "build_tokenizer",
    "get_fineweb_dataloader",
    "get_memmap_dataloader",
]
