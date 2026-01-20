import regex as re
import os
import logging
import time
import json
from multiprocessing import Pool

from collections import Counter, defaultdict
from tqdm import tqdm

# Precompiled regex pattern for pretokenization (GPT-2 style)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
REGEX_PATTERN = re.compile(PAT)
DEFAULT_NUM_WORKERS = 4

# Global timing variables for profiling (used in demo/main)
_GLOBAL_TIMING = {
    "merge_substeps": {
        "init_pair_counter": 0.0,
        "find_frequent_pair": 0.0,
        "merge_pairs": 0.0,
        "other_overhead": 0.0,
        "total": 0.0,
    }
}


def init_vocab(special_tokens: list[str]) -> list[bytes]:
    vocab = [token.encode("utf-8") for token in special_tokens]
    vocab.extend([bytes([i]) for i in range(256)])
    return vocab


def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_bytes: bytes = "<|endoftext|>".encode("utf-8"),
) -> list[int]:
    total_size = os.path.getsize(file_path)
    logging.info(
        f"Finding chunk boundaries: {file_path} ({total_size:,} bytes, target: {desired_num_chunks} chunks)"
    )

    with open(file_path, "rb") as f:
        # Get total file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        # Adjust boundaries to align with document boundaries
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size)

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_bytes)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique
        boundaries = sorted(set(chunk_boundaries))

    logging.info(f"Finished finding boundaries: {len(boundaries) - 1} chunks created")
    return boundaries


def _pretokenize_chunk_worker(args: tuple[str, int, int]) -> Counter[tuple[bytes, ...]]:
    file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        text = chunk.decode("utf-8", errors="ignore")

        # Split by special token to get individual documents
        # The special token itself should not be pretokenized
        documents = text.split("<|endoftext|>")

        # Pretokenize each document separately and aggregate
        total_counter = Counter()
        for doc in documents:
            if doc:  # Skip empty strings
                total_counter.update(pretokenize(doc))

        return total_counter


def pretokenize_chunks(
    file_path: str, boundaries: list[int], num_workers: int | None = None
) -> Counter[tuple[bytes, ...]]:
    """Pretokenize file by processing byte ranges in parallel."""
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    num_chunks = len(boundaries) - 1
    logging.info(
        f"Starting pretokenization of {num_chunks} chunks with {num_workers} workers"
    )

    if num_chunks == 0:
        return Counter()

    # Create list of (file_path, start, end) tuples for each chunk
    chunk_args = [
        (file_path, boundaries[i], boundaries[i + 1]) for i in range(num_chunks)
    ]

    total_counter = Counter()
    with Pool(processes=num_workers) as pool:
        for counter in pool.imap_unordered(_pretokenize_chunk_worker, chunk_args):
            total_counter.update(counter)

    logging.info(
        f"Finished pretokenization: {len(total_counter)} unique token sequences"
    )
    return total_counter


def pretokenize(
    text: str, regex_pattern: re.Pattern = REGEX_PATTERN
) -> Counter[tuple[bytes, ...]]:
    return Counter(
        tuple(bytes([b]) for b in match.group().encode("utf-8"))
        for match in regex_pattern.finditer(text)
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
    other_overhead = total_time - init_time - find_pair_time - merge_counter_time

    # Store timing in global variable for profiling
    _GLOBAL_TIMING["merge_substeps"] = {
        "init_pair_counter": init_time,
        "find_frequent_pair": find_pair_time,
        "merge_pairs": merge_counter_time,
        "other_overhead": other_overhead,
        "total": total_time,
    }

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
    logging.info(f"    - Other overhead: {other_overhead:.2f}s")
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
    return max(pair_counter, key=lambda pair: (pair_counter[pair], pair))


def merge_most_frequent_pair_in_counter(
    most_frequent_pair: tuple[bytes, bytes],
    pair_counter: Counter[tuple[bytes, bytes]],
    reversed_index: defaultdict[tuple[bytes, bytes], set[int]],
    id_to_tokens: dict[int, tuple[bytes, ...]],
    id_to_count: dict[int, int],
) -> None:
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

            if new_count < 0:
                raise ValueError(
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
    logging.info("=" * 80)
    logging.info(f"Starting BPE training")
    logging.info(f"  Input: {input_path}")
    logging.info(f"  Target vocab size: {vocab_size}")
    logging.info(f"  Special tokens: {special_tokens}")

    training_start = time.time()
    timing_breakdown = {}

    # Step 1: Find chunk boundaries (divides file into byte ranges)
    step_start = time.time()
    num_chunks = os.cpu_count() or 1  # Use number of CPU cores
    boundaries = find_chunk_boundaries(input_path, num_chunks)
    timing_breakdown["chunking"] = time.time() - step_start
    logging.info(f"[STEP 1/4] Chunking completed ({timing_breakdown['chunking']:.2f}s)")

    # Step 2: Initialize vocab
    step_start = time.time()
    vocab = init_vocab(special_tokens)
    timing_breakdown["vocab_init"] = time.time() - step_start
    logging.info(
        f"[STEP 2/4] Vocab initialization completed (initial vocab size: {len(vocab)}, {timing_breakdown['vocab_init']:.2f}s)"
    )

    # Step 3: Pretokenize
    step_start = time.time()
    pretokens_counter = pretokenize_chunks(input_path, boundaries)
    timing_breakdown["pretokenization"] = time.time() - step_start
    logging.info(
        f"[STEP 3/4] Pretokenization completed ({timing_breakdown['pretokenization']:.2f}s)"
    )

    # Step 4: Merge tokens
    step_start = time.time()
    num_merges = vocab_size - len(special_tokens) - 256
    vocab, merges = merge_token_pairs(vocab, pretokens_counter, num_merges)
    timing_breakdown["token_merging"] = time.time() - step_start
    logging.info(
        f"[STEP 4/4] Token merging completed ({timing_breakdown['token_merging']:.2f}s)"
    )

    total_time = time.time() - training_start

    # Print timing summary
    logging.info("=" * 80)
    logging.info(f"BPE training completed (final vocab size: {len(vocab)})")
    logging.info(f"")
    logging.info(f"TIMING SUMMARY:")
    logging.info(
        f"  Step 1 - Chunking:          {timing_breakdown['chunking']:>8.2f}s ({timing_breakdown['chunking']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"  Step 2 - Vocab Init:        {timing_breakdown['vocab_init']:>8.2f}s ({timing_breakdown['vocab_init']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"  Step 3 - Pretokenization:   {timing_breakdown['pretokenization']:>8.2f}s ({timing_breakdown['pretokenization']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"  Step 4 - Token Merging:     {timing_breakdown['token_merging']:>8.2f}s ({timing_breakdown['token_merging']/total_time*100:>5.1f}%)"
    )

    # Print merge substep breakdown from global timing
    merge_timing = _GLOBAL_TIMING["merge_substeps"]
    logging.info(
        f"    ├─ Init pair counter:     {merge_timing['init_pair_counter']:>8.2f}s ({merge_timing['init_pair_counter']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"    ├─ Find frequent pair:    {merge_timing['find_frequent_pair']:>8.2f}s ({merge_timing['find_frequent_pair']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"    ├─ Merge pairs:           {merge_timing['merge_pairs']:>8.2f}s ({merge_timing['merge_pairs']/total_time*100:>5.1f}%)"
    )
    logging.info(
        f"    └─ Other overhead:        {merge_timing['other_overhead']:>8.2f}s ({merge_timing['other_overhead']/total_time*100:>5.1f}%)"
    )

    logging.info(f"  {'─' * 50}")
    logging.info(f"  Total Training Time:        {total_time:>8.2f}s (100.0%)")
    logging.info("=" * 80)

    return {i: token for i, token in enumerate(vocab)}, merges


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    # Create output directory
    output_dir = "./out"
    os.makedirs(output_dir, exist_ok=True)

    # Save vocab as JSON (convert bytes to base64 strings for JSON serialization)
    vocab_path = os.path.join(output_dir, "tokenizer.json")
    vocab_serializable = {
        str(idx): token.decode("utf-8", errors="replace")
        for idx, token in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    logging.info(f"Vocab saved to {vocab_path}")

    # Save merges as text file (one merge per line, similar to GPT-2 format)
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            # Decode bytes to string, replacing any invalid UTF-8 with replacement character
            token1_str = token1.decode("utf-8", errors="replace")
            token2_str = token2.decode("utf-8", errors="replace")
            f.write(f"{token1_str} {token2_str}\n")
    logging.info(f"Merges saved to {merges_path}")


if __name__ == "__main__":
    main()
