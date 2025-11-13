import os
import json
import argparse
from tqdm import tqdm


def split_jsonl_by_lines(input_path, output_dir, lines_per_file=1000, num_files=8):
    os.makedirs(output_dir, exist_ok=True)

    total_needed = lines_per_file * num_files
    
    # Read and collect all entries first (you can stream if memory is tight)
    with open(input_path, 'r') as f:
        lines = [json.loads(line) for i, line in enumerate(f)]

    print(len(lines))

    total_lines = min(len(lines), total_needed)

    for i in range(num_files):
        start = i * lines_per_file
        end = min(start + lines_per_file, total_lines)
        if start >= end:
            break
        output_path = os.path.join(output_dir, f"sample_dclm_{i}.jsonl")
        with open(output_path, 'w') as out_f:
            for entry in lines[start:end]:
                out_f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL into 8 files with 1000 lines each")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    split_jsonl_by_lines(
        input_path=args.input_path,
        output_dir=args.output_dir,
        lines_per_file=20000,
        num_files=16
    )