"""Tokenize a raw text file into a flat int32 binary file loadable by np.memmap.

Usage:
    python -m scripts.preprocess out-tinystory data/TinyStoriesV2-GPT4-train.txt
    python -m scripts.preprocess out-tinystory data/TinyStoriesV2-GPT4-train.txt --workers 16
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

from answer.tokenizer import Tokenizer

CHUNK_SIZE = 1024 * 1024  # ~1 MB of text per chunk
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]

_worker_tokenizer: Tokenizer | None = None


def _worker_init(vocab_path: str, merges_path: str, special_tokens: list[str] | None):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def _worker_encode(text: str) -> bytes:
    return np.array(_worker_tokenizer.encode(text), dtype=np.int32).tobytes()


def read_chunks(input_path: str, chunk_size: int = CHUNK_SIZE):
    """Yield text chunks split on newline boundaries."""
    with open(input_path, "r", encoding="utf-8") as f:
        leftover = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if leftover:
                    yield leftover
                break
            text = leftover + chunk
            last_nl = text.rfind("\n")
            if last_nl == -1:
                leftover = text
                continue
            yield text[: last_nl + 1]
            leftover = text[last_nl + 1 :]


def tokenize_file(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None,
    input_path: str,
    output_path: str,
    num_workers: int,
) -> int:
    file_size = os.path.getsize(input_path)
    estimated_chunks = max(1, file_size // CHUNK_SIZE)
    total_tokens = 0
    chunks_done = 0

    with mp.Pool(num_workers, initializer=_worker_init, initargs=(vocab_path, merges_path, special_tokens)) as pool:
        with open(output_path, "wb") as fout:
            for encoded in pool.imap(_worker_encode, read_chunks(input_path)):
                fout.write(encoded)
                total_tokens += len(encoded) // 4
                chunks_done += 1
                pct = min(100, chunks_done * 100 // estimated_chunks)
                print(f"\r  {pct:3d}% — {total_tokens:,} tokens so far", end="", flush=True)

    print(f"\r  100% — {total_tokens:,} tokens total        ")
    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text to int32 binary for training"
    )
    parser.add_argument(
        "tokenizer_dir",
        type=str,
        help="Directory containing tokenizer.json and merges.txt",
    )
    parser.add_argument("input", type=str, help="Path to input .txt file")
    parser.add_argument(
        "--output",
        type=str,
        default="./training_data/tokens.npy",
        help="Path to output file (default: ./training_data/tokens.npy)",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=None,
        help="Special tokens (default: ['<|endoftext|>'])",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=f"Number of worker processes (default: 8)",
    )
    args = parser.parse_args()

    special_tokens = (
        args.special_tokens
        if args.special_tokens is not None
        else DEFAULT_SPECIAL_TOKENS
    )

    vocab_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    merges_path = os.path.join(args.tokenizer_dir, "merges.txt")

    for p in [vocab_path, merges_path, args.input]:
        if not os.path.exists(p):
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading tokenizer from {args.tokenizer_dir} ...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens or None)
    print(f"  vocab size: {len(tokenizer.vocab)}, merges: {len(tokenizer.merges)}")

    print(f"Tokenizing {args.input} -> {args.output} ({args.workers} workers) ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n = tokenize_file(
        vocab_path,
        merges_path,
        special_tokens or None,
        args.input,
        args.output,
        args.workers,
    )

    out_size = os.path.getsize(args.output)
    in_size = os.path.getsize(args.input)
    print(
        f"Done. {n:,} tokens, {out_size:,} bytes (compression: {in_size / n:.2f} bytes/token)"
    )


if __name__ == "__main__":
    main()
