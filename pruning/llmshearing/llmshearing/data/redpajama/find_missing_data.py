import os
import re

# Set your target directory
directory = "/scratch/yx3038/Research/pruning/LLM-Shearing/llmshearing/data/orig_data/for_prune/cc"

# Regular expression to match the shard filenames
pattern = re.compile(r"shard\.(\d{5})\.mds")

# Extract all shard numbers
shard_ids = []
for filename in os.listdir(directory):
    match = pattern.fullmatch(filename)
    if match:
        shard_ids.append(int(match.group(1)))

if not shard_ids:
    print("No valid shard files found.")
    exit(1)

# Detect missing shard IDs
shard_ids.sort()
expected = set(range(shard_ids[0], shard_ids[-1] + 1))
actual = set(shard_ids)
missing = sorted(expected - actual)

# Report
if missing:
    print("Missing shard files:")
    for i in missing:
        print(f"shard.{i:05d}.mds")
else:
    print("No missing files detected.")