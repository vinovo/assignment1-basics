#!/usr/bin/env python3
"""
Script to sample 10 random documents from a file delimited by <|endoftext|>
and write them to an output file.

Usage:
    python sample.py <input_file> <output_file>

Example:
    python sample.py data/TinyStoriesV2-GPT4-train.txt tmp/sampled_stories.txt
"""

import sys
import random
from pathlib import Path


def sample_documents(input_file: str, output_file: str, num_samples: int = 10, seed: int = 42):
    """
    Sample random documents from an input file and write to output file.
    Uses reservoir sampling for memory efficiency with large files.
    
    Args:
        input_file: Path to input file with documents delimited by <|endoftext|>
        output_file: Path to output file where sampled documents will be written
        num_samples: Number of documents to sample (default: 10)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    print(f"Reading documents from {input_file}...")
    
    delimiter = '<|endoftext|>'
    
    # Use reservoir sampling to sample documents without loading entire file
    # This maintains uniform random sampling while being memory efficient
    reservoir = []
    doc_count = 0
    current_doc = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Check if line contains the delimiter
            if delimiter in line:
                # Split by delimiter to handle multiple delimiters in one line
                parts = line.split(delimiter)
                
                for i, part in enumerate(parts):
                    if i == 0:
                        # Add to current document
                        current_doc.append(part)
                    
                    # Process the completed document (before this delimiter)
                    if current_doc:
                        doc_text = ''.join(current_doc).strip()
                        if doc_text:  # Only count non-empty documents
                            doc_count += 1
                            
                            # Reservoir sampling algorithm
                            if len(reservoir) < num_samples:
                                reservoir.append(doc_text)
                            else:
                                # Randomly replace elements with decreasing probability
                                j = random.randint(0, doc_count - 1)
                                if j < num_samples:
                                    reservoir[j] = doc_text
                        
                        current_doc = []
                    
                    # Start new document with remaining part (if not last part)
                    if i < len(parts) - 1:
                        current_doc = [part] if i > 0 else []
                    elif i == len(parts) - 1 and part:
                        # Last part after delimiter becomes start of next doc
                        current_doc = [part]
            else:
                current_doc.append(line)
    
    # Handle last document if file doesn't end with delimiter
    if current_doc:
        doc_text = ''.join(current_doc).strip()
        if doc_text:
            doc_count += 1
            if len(reservoir) < num_samples:
                reservoir.append(doc_text)
            else:
                j = random.randint(0, doc_count - 1)
                if j < num_samples:
                    reservoir[j] = doc_text
    
    print(f"Found {doc_count} documents in total")
    print(f"Sampled {len(reservoir)} documents")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write sampled documents to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(reservoir):
            f.write(doc)
            # Add delimiter after each document except the last one
            if i < len(reservoir) - 1:
                f.write(f"\n{delimiter}\n")
    
    print(f"Successfully wrote {len(reservoir)} documents to {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python sample.py <input_file> <output_file>")
        print("Example: python sample.py data/TinyStoriesV2-GPT4-train.txt tmp/sampled_stories.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    sample_documents(input_file, output_file)


if __name__ == "__main__":
    main()
