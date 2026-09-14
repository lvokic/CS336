"""Encode a text corpus to a compact uint16 token-id stream.

Example:
    uv run --no-sync python experiments/prepare_tokens.py \\
      --tokenizer data/tinystories_tokenizer.bin \\
      --input data/TinyStoriesV2-GPT4-train.txt \\
      --output data/tinystories_train_tokens.bin
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer, _iter_file_chunks


_worker_tokenizer: Tokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    """Load one immutable tokenizer in each encoder process."""
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.load(tokenizer_path)


def _encode_chunk(chunk: str) -> np.ndarray:
    """Encode an independently safe chunk in a worker process."""
    if _worker_tokenizer is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return np.asarray(_worker_tokenizer.encode(chunk), dtype="<u2")


def write_token_ids(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    input_path: Path,
    output_path: Path,
    chunk_bytes: int,
    workers: int,
) -> int:
    """Parallel-encode ``input_path`` and atomically write uint16 IDs.

    The chunk iterator only returns boundaries that preserve pretokenization,
    including whitespace runs and special tokens.  Therefore each process can
    encode a chunk independently. ``ProcessPoolExecutor.map`` yields results
    in input order, preserving the original token stream exactly.
    """
    if len(tokenizer.vocab) > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 cannot represent this tokenizer's vocabulary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    count = 0
    try:
        with temporary.open("wb") as destination:
            chunks = _iter_file_chunks(
                input_path,
                chunk_bytes,
                tuple(tokenizer.special_tokens),
            )
            if workers == 1:
                encoded_chunks = (
                    np.asarray(tokenizer.encode(chunk), dtype="<u2")
                    for chunk in chunks
                )
                for chunk_index, token_ids in enumerate(encoded_chunks, start=1):
                    token_ids.tofile(destination)
                    count += len(token_ids)
                    print(f"encoded chunk {chunk_index}: {count:,} tokens", flush=True)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(str(tokenizer_path),),
                ) as executor:
                    for chunk_index, token_ids in enumerate(
                        executor.map(_encode_chunk, chunks, chunksize=1), start=1
                    ):
                        token_ids.tofile(destination)
                        count += len(token_ids)
                        print(f"encoded chunk {chunk_index}: {count:,} tokens", flush=True)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a corpus into uint16 token IDs.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    args = parser.parse_args()
    if args.chunk_bytes <= 0:
        parser.error("--chunk-bytes must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    tokenizer = Tokenizer.load(args.tokenizer)
    token_count = write_token_ids(
        tokenizer,
        args.tokenizer,
        args.input,
        args.output,
        args.chunk_bytes,
        args.workers,
    )
    token_ids = np.memmap(args.output, dtype="<u2", mode="r")
    metadata = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "dtype": "uint16",
        "token_count": token_count,
        "vocab_size": len(tokenizer.vocab),
        "min_token_id": int(token_ids.min()) if token_count else None,
        "max_token_id": int(token_ids.max()) if token_count else None,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {token_count:,} tokens to {args.output} "
        f"(ID range: {metadata['min_token_id']}..{metadata['max_token_id']})"
    )


if __name__ == "__main__":
    main()
