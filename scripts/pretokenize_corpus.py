#!/usr/bin/env python3
"""Pre-tokenize a Hugging Face text corpus into flat memmap ``*.bin`` shards.

Run this on a machine WITH plenty of RAM/disk/network (desktop, workstation,
cloud instance).  Copy the output folder to your small training laptop and
train with ``--local_tokens_dir <folder>`` -- the laptop never downloads,
parses or tokenizes anything, and the reader RAM cost is ~zero.

Example (on the big machine)::

    python scripts/pretokenize_corpus.py \
        --out_dir ./fineweb_tokens \
        --dataset_name HuggingFaceFW/fineweb-edu \
        --tokenizer gpt2 \
        --train_tokens 1_000_000_000 \
        --val_tokens 5_000_000 \
        --tokens_per_shard 100_000_000

Then on the laptop::

    python train.py --variant 100m --precision fp4 --batch_size 2 \
        --grad_accum 32 --seq_len 2048 --grad_checkpoint --low_ram \
        --local_tokens_dir ./fineweb_tokens --disable_wandb

Output layout::

    out_dir/
        meta.json          # vocab_size, dtype, tokenizer, shard manifest
        val-00000.bin      # first --val_tokens tokens (validation)
        train-00000.bin    # remaining tokens, split into shards
        train-00001.bin
        ...

Tokens are stored as flat little-endian uint16 when the vocabulary fits
(gpt2: 50257, char256: 257) and uint32 otherwise (cl100k_base: 100277).
Documents are separated by the tokenizer's end-of-text id.  Shards are
written with EXACTLY tokens_per_shard tokens (the final shard may be
shorter); token ids themselves are never split across shards beyond the
needed rotation, which is fine -- training windows are segment-aligned at
load time anyway.  RAM stays constant: parquet shards are streamed one
row-group at a time and token ids are written in bounded chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmythos.dataset import build_tokenizer   # noqa: E402

log = logging.getLogger("pretokenize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-tokenize an HF corpus into OpenMythos memmap shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", required=True,
                   help="output folder for *.bin shards + meta.json")
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="gpt2",
                   choices=["gpt2", "cl100k_base", "char256"],
                   help="tiktoken encoding to use (must match training)")
    p.add_argument("--train_tokens", type=int, default=1_000_000_000,
                   help="training tokens to write after the val slice")
    p.add_argument("--val_tokens", type=int, default=5_000_000,
                   help="tokens reserved for the validation split")
    p.add_argument("--tokens_per_shard", type=int, default=100_000_000,
                   help="rotate to a new *.bin file every N tokens")
    p.add_argument("--max_parquet_shards", type=int, default=64,
                   help="cap on source parquet shards (~100-300 MB each)")
    p.add_argument("--data_mode", choices=["native", "stream"], default="native",
                   help="native: hub download + local pyarrow (robust); "
                        "stream: live load_dataset(streaming=True)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--flush_chunk_tokens", type=int, default=1 << 16,
                   help="tokens buffered in memory before each disk write")
    return p.parse_args()


class ShardWriter:
    """Bounded-buffer binary shard writer with EXACT-capacity rotation.

    Each closed shard holds exactly ``tokens_per_shard`` tokens; only the
    final shard of a split may be shorter.  Total tokens written never
    exceeds :meth:`room` remaining capacity, so callers can blind-append a
    stream without overshooting a target split size.
    """

    def __init__(self, out_dir: str, prefix: str, cap_tokens: int,
                 tokens_per_shard: int, np_dtype, flush_chunk: int):
        import numpy as np

        self._np = np
        self.out_dir = out_dir
        self.prefix = prefix
        self.cap_tokens = int(cap_tokens)            # hard stop for this split
        self.tokens_per_shard = max(int(tokens_per_shard), 1)
        self.np_dtype = np_dtype
        self.itemsize = int(np.dtype(np_dtype).itemsize)
        self.flush_chunk = max(int(flush_chunk), 1)
        self.shard_index = 0
        self.written_in_shard = 0
        self.written_total = 0
        self.files: List[str] = []
        self._fh = None
        self._buf: list = []
        self._open_next()

    def room(self) -> int:
        """Tokens this split will still accept."""
        return max(self.cap_tokens - self.written_total, 0)

    @property
    def full(self) -> bool:
        return self.written_total >= self.cap_tokens

    def _open_next(self) -> None:
        if self._fh is not None:
            self._fh.close()
        name = f"{self.prefix}-{self.shard_index:05d}.bin"
        self._fh = open(os.path.join(self.out_dir, name), "wb")
        self.files.append(name)
        self.written_in_shard = 0
        self.shard_index += 1
        log.info("opened shard %s/%s", self.out_dir, name)

    def append(self, ids) -> int:
        """Write up to :meth:`room` ids; return the number consumed."""
        if self.full:
            return 0
        take = min(len(ids), self.room())
        self._buf.extend(int(i) for i in ids[:take])
        self.written_total += take
        self._drain()
        return take

    def _drain(self) -> None:
        while self._buf:
            room = self.tokens_per_shard - self.written_in_shard
            chunk = self._buf[:room]
            arr = self._np.asarray(chunk, dtype=self.np_dtype)
            self._fh.write(arr.tobytes())
            del self._buf[:room]
            self.written_in_shard += len(chunk)
            if self.written_in_shard >= self.tokens_per_shard and self._buf:
                self._open_next()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        # Remove an entirely empty trailing shard (rotation opened but no
        # tokens ever landed in it).
        if self.written_in_shard == 0 and len(self.files) > 1:
            empty = os.path.join(self.out_dir, self.files.pop())
            os.remove(empty)
            self.shard_index -= 1


def _token_stream(args, tokenizer, eos_id: int):
    """Yield token-id lists from a FineWebStreamDataset's raw doc iterator."""
    from openmythos.dataset import FineWebStreamDataset

    ds = FineWebStreamDataset(
        dataset_name=args.dataset_name,
        split=args.split,
        tokenizer_name=args.tokenizer,
        seq_len=2048,                  # unused by the raw doc path
        seed=args.seed,
        shuffle_buffer_docs=1,        # pre-tokenization needs no reservoir
        streaming_shuffle=False,
        low_ram_profile=True,
        tokenize_chunk_docs=1 << 30,  # pacing is irrelevant (no DataLoader)
        tokenize_pause_s=0.0,
        data_mode=args.data_mode,
        max_parquet_shards=args.max_parquet_shards,
    )
    for doc in ds._raw_doc_iter(epoch=0):
        if isinstance(doc, dict) and "text_tokens" in doc:
            tokens = list(doc["text_tokens"])
        else:
            text = doc.get("text", "") if isinstance(doc, dict) else ""
            if not text:
                continue
            tokens = list(tokenizer.encode(text))
        tokens.append(eos_id)
        yield tokens


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    args = parse_args()
    import numpy as np

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, vocab_size = build_tokenizer(args.tokenizer)
    if hasattr(tokenizer, "eot_token"):
        eos_id = int(tokenizer.eot_token)
    elif hasattr(tokenizer, "eos_token"):
        eos_id = int(tokenizer.eos_token)
    else:
        eos_id = min(256, vocab_size - 1)   # char256 pad lane
    np_dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    log.info("tokenizer=%s vocab=%d eos=%d dtype=%s",
             args.tokenizer, vocab_size, eos_id, np.dtype(np_dtype).name)

    # Val shards are kept small individually (val is usually tiny); train
    # shards rotate at the requested chunk size.
    val_writer: Optional[ShardWriter] = None
    if args.val_tokens > 0:
        val_shard_size = min(args.tokens_per_shard, args.val_tokens)
        val_writer = ShardWriter(out_dir, "val", args.val_tokens,
                                 val_shard_size, np_dtype,
                                 args.flush_chunk_tokens)
    train_writer = ShardWriter(out_dir, "train", args.train_tokens,
                               args.tokens_per_shard, np_dtype,
                               args.flush_chunk_tokens)

    t0 = time.time()
    seen = 0
    last_log = 0
    try:
        for tokens in _token_stream(args, tokenizer, eos_id):
            if train_writer.full and (val_writer is None or val_writer.full):
                break
            offset = 0
            while offset < len(tokens):
                if val_writer is not None and not val_writer.full:
                    consumed = val_writer.append(tokens[offset:])
                    offset += consumed
                    if val_writer.full:
                        val_writer.close()
                        log.info("validation split complete: %d tokens",
                                 val_writer.written_total)
                    continue
                if train_writer.full:
                    break
                consumed = train_writer.append(tokens[offset:])
                offset += consumed
                if consumed == 0:
                    break
            seen = train_writer.written_total + (
                val_writer.written_total if val_writer is not None else 0)
            if seen - last_log >= 5_000_000:
                last_log = seen
                rate = seen / max(time.time() - t0, 1e-6)
                log.info("progress: %.1fM tokens (%.0f tok/s)",
                         seen / 1e6, rate)
    finally:
        if val_writer is not None:
            val_writer.close()
        train_writer.close()

    val_count = val_writer.written_total if val_writer is not None else 0
    files = (val_writer.files if val_writer is not None else []) \
        + train_writer.files
    manifest = {
        "dataset_name": args.dataset_name,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "dtype": np.dtype(np_dtype).name,
        "train_tokens": train_writer.written_total,
        "val_tokens": val_count,
        "files": files,
        "created_unix": int(time.time()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("done: %d train + %d val tokens in %s",
             train_writer.written_total, val_count, out_dir)
    log.info("copy this folder to the training laptop and run train.py with "
             "--local_tokens_dir %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
