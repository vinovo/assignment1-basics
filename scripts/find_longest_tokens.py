import json

# Load the tokenizer
with open("owt-out/tokenizer.json", "r") as f:
    tokenizer = json.load(f)

# Create a list of (token_id, token_string, length) tuples
token_lengths = []
for token_id, token_string in tokenizer.items():
    token_lengths.append((token_id, token_string, len(token_string)))

# Sort by length in descending order and get top 5
top_5 = sorted(token_lengths, key=lambda x: x[2], reverse=True)[:5]

# Print results
print("Top 5 longest tokens:\n")
for i, (token_id, token_string, length) in enumerate(top_5, 1):
    print(f"{i}. Token ID: {token_id}")
    print(f"   Length: {length}")
    print(f"   Token: {repr(token_string)}")
    print()
