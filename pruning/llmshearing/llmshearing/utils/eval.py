import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import torch
from lm_eval import evaluator, tasks, models
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_FEWSHOTS = {
    "winogrande": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "arc_challenge_chat": 25,
    "hellaswag": 10,
    "sciq": 0,
    "piqa": 0
}

def run_eval(paths, tasks):
    results = {}
    for path in paths:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
        lm_eval_model = models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        for task in tasks:
        
            res = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[task],
                num_fewshot=NUM_FEWSHOTS[task],
                # num_fewshot=0,
                max_batch_size=64,
                apply_chat_template=False,
                log_samples=True,
            )
            
            print(res['results'])
        results[task] = res['results']
    return results

if __name__ == '__main__':
    # hf_model_path = '/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/Llama-2-2.7b-hf'
    # hf_model_path2 = '/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/Sheared-LLaMA-2.7B-Pruned'
    # hf_model_path='/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/Sheared-LLaMA-2.7B-Pruned'
    # results = run_eval(
    #     # paths=['/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/Sheared-LLaMA-1.3B'],
    #     paths=[hf_model_path, hf_model_path2],
    #     tasks=['arc_challenge', 'hellaswag']
    # )
        
    results = run_eval(
        paths=['/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/Sheared-LLaMA-1.3B-Pruned'],
        tasks=['sciq', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag']
    )
    print(results)