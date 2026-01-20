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

        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
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
        # Pass 1: Find all special token positions and split text
        special_token_positions = []  # List of (start_pos, end_pos, token)
        
        # Scan through text to find all special tokens
        pos = 0
        while pos < len(text):
            matched = False
            for special_token in self.special_tokens:
                if text[pos:pos+len(special_token)] == special_token:
                    special_token_positions.append((pos, pos + len(special_token), special_token))
                    pos += len(special_token)
                    matched = True
                    break
            if not matched:
                pos += 1
        
        # Split into text_parts and special_tokens_found
        # We'll have len(special_tokens_found) + 1 text parts
        text_parts = []
        special_tokens_found = []
        last_pos = 0
        
        for start, end, token in special_token_positions:
            text_parts.append(text[last_pos:start])  # Text before this special token (could be empty)
            special_tokens_found.append(token)
            last_pos = end
        text_parts.append(text[last_pos:])  # Text after last special token (could be empty)
        
        # Pass 2: Process each text part with regex, then interleave with special tokens
        result = []
        
        for i, text_part in enumerate(text_parts):
            # Process this text part with regex (skip if empty)
            if text_part:
                for match in self.PRETOKENIZE_REGEX_PATTERN.finditer(text_part):
                    matched_text = match.group()
                    result.append(tuple(bytes([b]) for b in matched_text.encode("utf-8")))
            
            # Add special token after this text part (if one exists)
            if i < len(special_tokens_found):
                result.append((special_tokens_found[i].encode("utf-8"),))
        
        return result

    def encode(self, text: str) -> list[int]:
        token_groups = self.pretokenize(text)

        return [
            self.reversed_vocab[token]
            for tokens in token_groups
            for token in self.merge_token(tokens)
        ]

    def merge_token(self, tokens: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # special tuples that hold special tokens
        if len(tokens) == 1 and tokens[0] in self.special_tokens:
            return tokens

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
        i = 0
        merge_indices = set()

        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == merge:
                merge_indices.add(i)
                i += 2
            else:
                i += 1

        if not merge_indices:
            return False, tokens

        # actualy apply the merges
        new_tokens = []
        i = 0

        while i < len(tokens) - 1:
            if i in merge_indices:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        if i == len(tokens) - 1:
            new_tokens.append(tokens[i])

        return True, tuple(new_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id_] for id_ in ids).decode("utf-8", errors="replace")
