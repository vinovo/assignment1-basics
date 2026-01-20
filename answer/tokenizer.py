import json
from typing import Iterable
import regex as re


def parse_merges_file(filepath: str) -> list[tuple[bytes, bytes]]:
    """
    Parse a merges file where each line contains two tokens separated by a space.
    
    The merge file uses Ġ (U+0120) as a replacement for spaces within tokens,
    so we can simply split by space and replace Ġ back to spaces.
    
    Args:
        filepath: Path to the merges file
        
    Returns:
        List of merge pairs as tuples of bytes
    """
    merges = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n\r')  # Remove line endings
            
            if not line:  # Skip empty lines
                continue
            
            # Split by space (the separator between the two tokens)
            parts = line.split(' ')
            if len(parts) != 2:
                raise ValueError(f"Could not parse merge line: {repr(line)} - expected exactly 2 tokens")
            
            token1_str, token2_str = parts
            
            # Replace Ġ back to spaces
            token1_str = token1_str.replace("Ġ", " ")
            token2_str = token2_str.replace("Ġ", " ")
            
            merges.append((token1_str.encode('utf-8'), token2_str.encode('utf-8')))
    
    return merges


# OLD COMPLEX LOGIC (commented out for reference):
# def parse_merges_file_old(filepath: str) -> list[tuple[bytes, bytes]]:
#     """
#     OLD: Parse a merges file where each line contains two tokens separated by a space.
#     
#     Since tokens can start with spaces (but spaces can only occur at the beginning),
#     we parse each line as:
#     {possible whitespace from token 1}{char part from token 1} {possible whitespace from token 2}{char part from token 2}
#     
#     Strategy: Find the first non-space character of token2 (scanning backwards),
#     then find the leftmost space in the sequence of spaces before it. This ensures
#     token2 gets any leading spaces per the pretokenization rules.
#     """
#     merges = []
#     with open(filepath, "r") as f:
#         for line in f:
#             line = line.rstrip('\n\r')
#             
#             if not line or len(line) <= 1:
#                 continue
#             
#             if line.strip() == '':
#                 split_pos = len(line) // 2
#                 token1 = line[:split_pos]
#                 token2 = line[split_pos + 1:]
#             else:
#                 last_non_space = -1
#                 for i in range(len(line) - 1, -1, -1):
#                     if line[i] != ' ':
#                         last_non_space = i
#                         break
#                 
#                 first_non_space_in_token2 = last_non_space
#                 for i in range(last_non_space, -1, -1):
#                     if line[i] == ' ':
#                         first_non_space_in_token2 = i + 1
#                         break
#                     if i == 0:
#                         raise ValueError(f"Could not parse merge line: {repr(line)}")
#                 
#                 split_pos = first_non_space_in_token2 - 1
#                 
#                 if split_pos < 0 or line[split_pos] != ' ':
#                     raise ValueError(f"Could not parse merge line: {repr(line)}")
#                 
#                 while split_pos > 0 and line[split_pos - 1] == ' ':
#                     split_pos -= 1
#                 
#                 if split_pos == 0:
#                     if first_non_space_in_token2 == 1:
#                         raise ValueError(f"Could not parse merge line: {repr(line)} - only one space before token2")
#                     split_pos = first_non_space_in_token2 - 1
#                 
#                 token1 = line[:split_pos]
#                 token2 = line[split_pos + 1:]
#             
#             merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
#     
#     return merges


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

    @classmethod
    def from_files(
        cls, vocav_filepath: str, merges_filepath: str, special_tokens: list[str] = None
    ):
        with open(vocav_filepath, "r") as f:
            vocab_json = json.load(f)
        
        # Convert vocab from JSON format (str keys, str values) to expected format (int keys, bytes values)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
        
        merges = parse_merges_file(merges_filepath)
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
