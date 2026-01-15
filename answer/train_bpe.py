import regex as re
import os
import logging
import time
from multiprocessing import Pool
import tempfile

from collections import Counter, defaultdict
from tqdm import tqdm

TMP_DIR = "tmp"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(special_tokens: list[str]) -> list[bytes]:
    vocab = [token.encode("utf-8") for token in special_tokens]
    vocab.extend([bytes([i]) for i in range(256)])
    return vocab


def chunk_file(
    file_path: str,
    split_bytes: bytes = "<|endoftext|>".encode("utf-8"),
    output_dir: str = TMP_DIR,
    BUFFER_SIZE: int = 8 * 1024 * 1024,  # 8MB
) -> list[str]:
    logging.info(f"Starting file chunking: {file_path}")

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

    logging.info(f"Finished file chunking: {len(written_files)} chunks created")
    return written_files


def _pretokenize_chunk_worker(chunk_path: str) -> Counter[tuple[bytes, ...]]:
    """Worker function to pretokenize a single chunk file."""
    with open(chunk_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
        return pretokenize(text)


def pretokenize_chunks(
    chunk_paths: list[str], num_workers: int | None = None
) -> Counter[tuple[bytes, ...]]:
    """Pretokenize multiple chunks in parallel and aggregate counters."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    logging.info(
        f"Starting pretokenization of {len(chunk_paths)} chunks with {num_workers} workers"
    )

    if not chunk_paths:
        return Counter()

    total_counter = Counter()
    with Pool(processes=num_workers) as pool:
        for counter in pool.imap_unordered(_pretokenize_chunk_worker, chunk_paths):
            total_counter.update(counter)

    logging.info(
        f"Finished pretokenization: {len(total_counter)} unique token sequences"
    )
    return total_counter


def pretokenize(text: str, regex_pattern: str = PAT) -> Counter[tuple[bytes, ...]]:
    return Counter(
        tuple(bytes([b]) for b in match.group().encode("utf-8"))
        for match in re.finditer(regex_pattern, text)
    )


def merge_token_pairs(
    vocab: list[bytes], counter: Counter[tuple[bytes, ...]], max_merges: int
) -> tuple[list[bytes], list[tuple[bytes, bytes]]]:
    logging.info(f"Starting token merging: {max_merges} merges to perform")

    total_start = time.time()

    # Time: Initialize pair counter and sequence ID mappings
    init_start = time.time()
    merges = []
    pair_counter, reversed_index, id_to_tokens, id_to_count = initialize_pair_counter(
        counter
    )
    init_time = time.time() - init_start

    # Accumulators for timing breakdown
    find_pair_time = 0.0
    merge_counter_time = 0.0

    for _ in tqdm(range(max_merges), desc="Merging tokens", unit="merge"):
        if not pair_counter:
            logging.info(f"No more pairs to merge after {len(merges)} merges")
            break

        # Time: Find most frequent pair
        find_start = time.time()
        most_frequent_pair = find_most_frequent_pair(pair_counter)
        find_pair_time += time.time() - find_start

        merged_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab.append(merged_token)
        merges.append(most_frequent_pair)

        # Time: Merge the pair using ID-based data structures
        merge_start = time.time()
        merge_most_frequent_pair_in_counter(
            most_frequent_pair, pair_counter, reversed_index, id_to_tokens, id_to_count
        )
        merge_counter_time += time.time() - merge_start

    total_time = time.time() - total_start

    # Print timing breakdown
    logging.info(f"Finished token merging: {len(merges)} merges completed")
    logging.info(f"  Timing breakdown:")
    logging.info(
        f"    - Initialize pair counter: {init_time:.2f}s ({init_time/total_time*100:.1f}%)"
    )
    logging.info(
        f"    - Find most frequent pair: {find_pair_time:.2f}s ({find_pair_time/total_time*100:.1f}%)"
    )
    logging.info(
        f"    - Merge pairs in counter: {merge_counter_time:.2f}s ({merge_counter_time/total_time*100:.1f}%)"
    )
    logging.info(
        f"    - Other overhead: {total_time - init_time - find_pair_time - merge_counter_time:.2f}s"
    )
    logging.info(f"    - Total: {total_time:.2f}s")

    return vocab, merges


def initialize_pair_counter(
    counter: Counter[tuple[bytes, ...]],
) -> tuple[
    Counter[tuple[bytes, bytes]],
    defaultdict[tuple[bytes, bytes], set[int]],
    dict[int, tuple[bytes, ...]],
    dict[int, int],
]:
    """
    Initialize pair counter and sequence ID mappings.

    Returns:
        - pair_counter: global frequency of each pair
        - reversed_index: maps each pair to set of sequence IDs containing it
        - id_to_tokens: maps sequence ID to token tuple
        - id_to_count: maps sequence ID to its count/frequency
    """
    pair_counter = Counter()
    reversed_index = defaultdict(set)
    id_to_tokens = {}
    id_to_count = {}

    # Assign each unique token sequence a sequential ID
    for seq_id, (tokens, count) in enumerate(counter.items()):
        id_to_tokens[seq_id] = tokens
        id_to_count[seq_id] = count

        # Build pair counter and reversed index using IDs
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counter[pair] += count
            reversed_index[pair].add(seq_id)

    return pair_counter, reversed_index, id_to_tokens, id_to_count


def find_most_frequent_pair(
    pair_counter: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    """
    Find the most frequent pair in the pair counter.

    Tie-breaking: Among pairs with equal frequency, selects the lexicographically
    largest pair (based on bytes ordering) for deterministic behavior.
    """
    return max(pair_counter, key=lambda pair: (pair_counter[pair], pair))


def merge_most_frequent_pair_in_counter(
    most_frequent_pair: tuple[bytes, bytes],
    pair_counter: Counter[tuple[bytes, bytes]],
    reversed_index: defaultdict[tuple[bytes, bytes], set[int]],
    id_to_tokens: dict[int, tuple[bytes, ...]],
    id_to_count: dict[int, int],
) -> None:
    """
    Merge the most frequent pair and update data structures accordingly.

    This function modifies pair_counter, reversed_index, and id_to_tokens in-place.
    id_to_count remains unchanged as sequence counts don't change during merges.
    """
    # Iterate over a snapshot to avoid "set changed size during iteration" errors
    affected_seq_ids = list(reversed_index[most_frequent_pair])

    for seq_id in affected_seq_ids:
        tokens = id_to_tokens[seq_id]
        count = id_to_count[seq_id]

        # Build new token sequence by merging all occurrences of most_frequent_pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_frequent_pair:
                merged_token = tokens[i] + tokens[i + 1]
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        new_tokens_tuple = tuple(new_tokens)

        # Extract old pairs and new pairs
        old_pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        new_pairs = [
            (new_tokens_tuple[i], new_tokens_tuple[i + 1])
            for i in range(len(new_tokens_tuple) - 1)
        ]

        # Count occurrences of each pair in old and new sequences
        old_pair_counts = Counter(old_pairs)
        new_pair_counts = Counter(new_pairs)

        # Update pair_counter: decrement old pairs, increment new pairs
        for pair, pair_count in old_pair_counts.items():
            old_count = pair_counter[pair]
            pair_counter[pair] -= pair_count * count
            new_count = pair_counter[pair]

            assert new_count >= 0, (
                f"Pair counter went negative for pair {pair}: "
                f"old_count={old_count}, subtracted={pair_count * count}, new_count={new_count}"
            )

            if pair_counter[pair] == 0:
                del pair_counter[pair]

        for pair, pair_count in new_pair_counts.items():
            pair_counter[pair] += pair_count * count

        # Update reversed_index: remove seq_id from old pairs, add to new pairs
        for pair in old_pair_counts:
            if pair in reversed_index:
                reversed_index[pair].discard(seq_id)
                # Clean up empty sets
                if not reversed_index[pair]:
                    del reversed_index[pair]

        for pair in new_pair_counts:
            reversed_index[pair].add(seq_id)

        # Update id_to_tokens with new token sequence
        id_to_tokens[seq_id] = new_tokens_tuple

    # Remove the merged pair from consideration
    if most_frequent_pair in pair_counter:
        del pair_counter[most_frequent_pair]
    if most_frequent_pair in reversed_index:
        del reversed_index[most_frequent_pair]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input text.
    """
    logging.info("=" * 80)
    logging.info(f"Starting BPE training")
    logging.info(f"  Input: {input_path}")
    logging.info(f"  Target vocab size: {vocab_size}")
    logging.info(f"  Special tokens: {special_tokens}")

    # Use temporary directory for chunk files that cleans up automatically
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Chunk file
        chunk_paths = chunk_file(input_path, output_dir=tmp_dir)
        logging.info(f"[STEP 1/4] Chunking completed")

        # Step 2: Initialize vocab
        vocab = init_vocab(special_tokens)
        logging.info(
            f"[STEP 2/4] Vocab initialization completed (initial vocab size: {len(vocab)})"
        )

        # Step 3: Pretokenize
        pretokens_counter = pretokenize_chunks(chunk_paths)
        logging.info(f"[STEP 3/4] Pretokenization completed")

    # Step 4: Merge tokens (temp directory cleaned up after pretokenization)
    num_merges = vocab_size - len(special_tokens) - 256
    vocab, merges = merge_token_pairs(vocab, pretokens_counter, num_merges)
    logging.info(f"[STEP 4/4] Token merging completed")

    logging.info(f"BPE training completed (final vocab size: {len(vocab)})")
    logging.info("=" * 80)

    return {i: token for i, token in enumerate(vocab)}, merges


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    vocab, merges = train_bpe(
        input_path="tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    print(vocab)
    print(merges)


if __name__ == "__main__":
    main()
