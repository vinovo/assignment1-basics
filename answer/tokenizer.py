import json
from typing import Iterable
import regex as re


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

        if special_tokens:
            for token in special_tokens:
                self.vocab[len(self.vocab)] = token.encode("utf-8")

        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        # Build pretokenization pattern with special tokens
        if special_tokens:
            # Escape special regex characters and join with |
            escaped_special = [re.escape(token) for token in special_tokens]
            special_pattern = "|".join(escaped_special)
            # Put special tokens first so they match before the general pattern
            full_pattern = f"({special_pattern})|{self.PAT}"
            self.pretokenize_regex = re.compile(full_pattern)
        else:
            self.pretokenize_regex = self.PRETOKENIZE_REGEX_PATTERN

        self.special_tokens = set(special_tokens)

    @classmethod
    def from_files(
        cls, vocav_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        with open(vocav_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = [tuple(line.strip().split().encode("utf-8")) for line in f]
        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> Iterable[list[tuple[bytes, ...]]]:
        result = []
        for match in self.pretokenize_regex.finditer(text):
            matched_text = match.group()
            # If this is a special token, keep it as a single unit
            if matched_text in self.special_tokens:
                result.append((matched_text.encode("utf-8"),))
            else:
                # Otherwise, break it into bytes
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
        pair_merge = True

        while pair_merge:
            pair_merge = False
            for merge in self.merges:
                pair_merge, tokens = self.try_merge_pair_on_tokens(tokens, merge)
                # we have at least found one merge
                if pair_merge:
                    break

        return tokens

    def try_merge_pair_on_tokens(
        self, tokens: tuple[bytes, ...], merge: tuple[bytes, bytes]
    ) -> tuple[bool, tuple[bytes, ...]]:
        merged_bytes = merge[0] + merge[1]
        i = j = 0
        merge_indices = []
        merge_start = 0

        # first iter, find all the indices to merge first
        while i < len(tokens) - 1:
            token = tokens[i]

            if token in self.special_tokens:
                # current match needs to stop when hitting a special token
                j = 0
                merge_start = i + 1
                i += 1
                continue

            if merged_bytes[j:].startswith(token):
                if j + len(token) == len(merged_bytes):
                    # found a merge, reset the pointers
                    merge_indices.append((merge_start, i + 1))
                    j = 0
                    merge_start = i + 1
                # not getting to the end of merged bytes yet
                else:
                    j += len(token)
            # if the match is not found
            else:
                j = 0
                merge_start = i + 1

            i += 1

        if not merge_indices:
            return False, tokens

        # actualy apply the merges
        new_tokens = []
        i = j = 0

        while i < len(tokens):
            if j < len(merge_indices) and i == merge_indices[j][0]:
                end_index = merge_indices[j][1]
                merged_token = b"".join(tokens[i:end_index])
                new_tokens.append(merged_token)
                i = end_index
                j += 1
            else:
                new_tokens.append(tokens[i])
                i += 1

        # assert all have been merged
        assert j == len(merge_indices)

        return True, new_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")
