#!/usr/bin/env python3
"""
Script to compute tokenization compression ratio.

Usage:
    python tokenize_exp.py <vocab_file> <merge_file> <text_file>

Example:
    python tokenize_exp.py out/tokenizer.json out/merges.txt data/TinyStoriesV2-GPT4-train.txt
"""

import sys
from pathlib import Path
from tokenizer import Tokenizer


def compute_compression_ratio(vocab_file: str, merge_file: str, text_file: str):
    print(f"Loading tokenizer from:")
    print(f"  Vocab: {vocab_file}")
    print(f"  Merges: {merge_file}")
    
    # Initialize tokenizer
    tokenizer = Tokenizer.from_files(vocab_file, merge_file)
    
    print(f"\nReading text from: {text_file}")
    
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Compute number of bytes
    num_bytes = len(text.encode('utf-8'))
    print(f"Text size: {num_bytes:,} bytes")
    
    # Encode the text
    print("Tokenizing...")
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    
    print(f"Number of tokens: {num_tokens:,}")
    
    # Compute compression ratio
    compression_ratio = num_bytes / num_tokens if num_tokens > 0 else 0
    
    print(f"\nCompression ratio: {compression_ratio:.4f} bytes/token")
    
    return compression_ratio


def main():
    if len(sys.argv) != 4:
        print("Usage: python tokenize_exp.py <vocab_file> <merge_file> <text_file>")
        print("Example: python tokenize_exp.py out/tokenizer.json out/merges.txt data/sample.txt")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    merge_file = sys.argv[2]
    text_file = sys.argv[3]
    
    # Check if files exist
    for filepath, name in [(vocab_file, "Vocabulary"), (merge_file, "Merge"), (text_file, "Text")]:
        if not Path(filepath).exists():
            print(f"Error: {name} file '{filepath}' does not exist")
            sys.exit(1)
    
    compute_compression_ratio(vocab_file, merge_file, text_file)


if __name__ == "__main__":
    main()
