import json
from transformers import AutoTokenizer

data_path = '/scratch/yx3038/Research/pruning/fms/llmshearing/data/sample_dclm/dclm/sample_dclm_0.jsonl'
tokenizer = AutoTokenizer.from_pretrained("/scratch/yx3038/model_ckpt/Llama-2-7b-hf/")

total_tokens = 0

with open(data_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        text = data["text"]
        tokens = tokenizer(text)["input_ids"]
        total_tokens += len(tokens)

print(f"Total tokens: {total_tokens:.2e}")