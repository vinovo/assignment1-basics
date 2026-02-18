import json
from typing import Iterable
import regex as re


def bytes_to_unicode() -> dict[int, str]:
    """Returns a mapping from byte values (0-255) to unique unicode characters,
    avoiding control characters and whitespace that would break serialization."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def bytes_to_str(token: bytes) -> str:
    """Encode a byte-level token to a safe unicode string."""
    return "".join(_BYTE_TO_UNICODE[b] for b in token)


def str_to_bytes(s: str) -> bytes:
    """Decode a safe unicode string back to bytes."""
    return bytes(_UNICODE_TO_BYTE[c] for c in s)


def parse_merges_file(filepath: str) -> list[tuple[bytes, bytes]]:
    """Parse a merges file where each line contains two tokens (as mapped unicode
    strings) separated by a space.

    Args:
        filepath: Path to the merges file

    Returns:
        List of merge pairs as tuples of bytes
    """
    merges = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if not line:
                continue

            parts = line.split(" ")
            if len(parts) != 2:
                raise ValueError(
                    f"Could not parse merge line: {repr(line)} - expected exactly 2 tokens"
                )

            token1_str, token2_str = parts
            merges.append((str_to_bytes(token1_str), str_to_bytes(token2_str)))

    return merges


class Tokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PRETOKENIZE_REGEX_PATTERN = re.compile(PAT)

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        special_tokens = special_tokens or []

        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        # Sort special tokens by length (longest first) to match longer tokens before shorter ones
        self.special_tokens = sorted(set(special_tokens), key=len, reverse=True)
        self.special_tokens_set = set(
            token.encode("utf-8") for token in self.special_tokens
        )

        if self.special_tokens:
            # Escape special regex chars and join with | (alternation)
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_token_regex = re.compile(pattern)
        else:
            self.special_token_regex = None

        self.merge_priority = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls, vocav_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        with open(vocav_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)

        vocab = {int(k): str_to_bytes(v) for k, v in vocab_json.items()}

        merges = parse_merges_file(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> Iterable[list[tuple[bytes, ...]]]:
        if self.special_token_regex:
            # Split text by special tokens while keeping the delimiters
            parts = self.special_token_regex.split(text)
            tokens = self.special_token_regex.findall(text)

            # Interleave text parts and special tokens
            result = []
            for i, text_part in enumerate(parts):
                if text_part:
                    for match in self.PRETOKENIZE_REGEX_PATTERN.finditer(text_part):
                        matched_text = match.group()
                        result.append(
                            tuple(bytes([b]) for b in matched_text.encode("utf-8"))
                        )

                # Add special token after this text part (if one exists)
                if i < len(tokens):
                    result.append((tokens[i].encode("utf-8"),))

            return result
        else:
            # No special tokens, just use regex pattern
            result = []
            for match in self.PRETOKENIZE_REGEX_PATTERN.finditer(text):
                matched_text = match.group()
                result.append(tuple(bytes([b]) for b in matched_text.encode("utf-8")))
            return result

    def encode(self, text: str) -> list[int]:
        token_groups = self.pretokenize(text)

        return [
            self.reversed_vocab[token]
            for tokens in token_groups
            for token in self.merge_token(tokens)
        ]

    def merge_token(self, tokens: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # special tokens should not be merged with other tokens
        if len(tokens) == 1 and tokens[0] in self.special_tokens_set:
            return tokens

        tokens = list(tokens)

        while True:
            # Find the highest priority (lowest index) merge that exists in tokens
            best_merge = None
            best_priority = len(self.merges)
            best_positions = []

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_merge = pair
                        best_positions = [i]
                    elif priority == best_priority:
                        best_positions.append(i)

            if best_merge is None:
                break

            # Apply this merge at all positions (non-overlapping)
            # Process from right to left to maintain indices
            merge_indices = set()
            for i in reversed(best_positions):
                # Check if this position is still valid (not overlapping with a previous merge)
                if i not in merge_indices and i + 1 not in merge_indices:
                    merge_indices.add(i)

            # Apply merges from right to left
            for i in sorted(merge_indices, reverse=True):
                tokens[i] = tokens[i] + tokens[i + 1]
                tokens.pop(i + 1)

        return tuple(tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id_] for id_ in ids).decode(
            "utf-8", errors="replace"
        )
