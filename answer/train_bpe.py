import re
import os
from multiprocessing import Pool

from collections import Counter
from typing import Any

TMP_DIR = "tmp"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]


def init_vocab() -> dict[int, bytes]:
    vocab = {i: token.encode("utf-8") for i, token in enumerate(SPECIAL_TOKENS)}

    offset = len(vocab)
    vocab.update({offset + i: bytes([i]) for i in range(256)})

    return vocab


def chunk_file(
    file_path: str,
    split_bytes: bytes = "<|endoftext|>".encode("utf-8"),
    output_dir: str = TMP_DIR,
    BUFFER_SIZE: int = 8 * 1024 * 1024,  # 8MB
) -> list[str]:

    def write_chunk(chunk: bytes, chunk_id: int) -> str:
        if chunk:  # Skip empty chunks
            file_name = f"chunk_{chunk_id}.txt"
            output_path = os.path.join(output_dir, file_name)
            output_path = os.path.abspath(output_path)
            with open(output_path, "wb") as out_f:
                out_f.write(chunk)
            return output_path

        return None

    os.makedirs(output_dir, exist_ok=True)
    written_files = []

    with open(file_path, "rb") as f:
        chunk_id = 0
        buffer = b""

        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break

            buffer += data
            special_token_index = buffer.find(split_bytes)

            while special_token_index != -1:
                chunk = buffer[:special_token_index]
                file_path = write_chunk(chunk, chunk_id)
                if file_path:
                    written_files.append(file_path)
                    chunk_id += 1

                buffer = buffer[special_token_index + len(split_bytes) :]
                special_token_index = buffer.find(split_bytes)

    if buffer:
        file_path = write_chunk(buffer, chunk_id)
        if file_path:
            written_files.append(file_path)
            chunk_id += 1

    return written_files


def _pretokenize_chunk_worker(chunk_path: str) -> Counter[bytes]:
    """Worker function to pretokenize a single chunk file."""
    with open(chunk_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
        return pretokenize(text)


def pretokenize_chunks(
    chunk_paths: list[str], num_workers: int = os.cpu_count()
) -> Counter[bytes]:
    """Pretokenize multiple chunks in parallel and aggregate counters."""
    if not chunk_paths:
        return Counter[bytes]()

    # Use multiprocessing to parallelize pretokenization
    total_counter = Counter[bytes]()
    with Pool(processes=num_workers) as pool:
        # imap_unordered yields results as they complete (no ordering needed)
        for counter in pool.imap_unordered(_pretokenize_chunk_worker, chunk_paths):
            total_counter.update(counter)

    return total_counter


def pretokenize(text: str, regex_pattern: str = PAT) -> Counter[bytes]:
    return Counter[bytes](
        match.group().encode("utf-8") for match in re.finditer(regex_pattern, text)
    )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input text.
    """
    chunk_paths = chunk_file(input_path)
