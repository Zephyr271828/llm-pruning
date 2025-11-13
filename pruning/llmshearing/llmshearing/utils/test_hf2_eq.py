import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmshearing.models.composer_llama import ComposerMosaicLlama
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def test_two_matrix(a, b, desc=""):
    """ test if two matrix are equal """
    s1 = a.sum().item(); s2 = b.sum().item() if b is not None else torch.tensor(0).to(a.device).to(a.dtype)
    try:
        assert abs(s1 - s2) < 1e-3
    except:
        print(f"[{desc}] failed! sums are not equal: {s1} vs {s2}")
        return 
    print(f"[{desc}] passed! sums are equal: {s1} vs {s2}")

if __name__ == "__main__":
    import sys
    
    hf_llama2_path = sys.argv[1]
    hf_llama2_path2 = sys.argv[2]
    
    tokenizer = AutoTokenizer.from_pretrained(hf_llama2_path)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Slack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")


    # check if they have the same naming convention
    # with init_empty_weights():
    #     hf_model_1 = AutoModelForCausalLM.from_pretrained(hf_llama2_path)
    # hf_model_1 = load_checkpoint_and_dispatch(
    #     hf_model_1,
    #     checkpoint=os.path.join(hf_llama2_path, 'model.safetensors.index.json'),  
    #     device_map="auto",          
    #     no_split_module_classes=["LlamaDecoderLayer"],  
    #     dtype=torch.bfloat16,       
    # )  
    hf_model_1 = AutoModelForCausalLM.from_pretrained(hf_llama2_path)
    with torch.no_grad():
        hf_loss_1 = hf_model_1(input_ids, labels=input_ids).loss
    
    # with init_empty_weights():
    #     hf_model_2 = AutoModelForCausalLM.from_pretrained(hf_llama2_path2)
    # hf_model_2 = load_checkpoint_and_dispatch(
    #     hf_model_2,
    #     checkpoint=os.path.join(hf_llama2_path2, 'model.safetensors.index.json'),  
    #     device_map="auto",          
    #     no_split_module_classes=["LlamaDecoderLayer"],  
    #     dtype=torch.bfloat16,       
    # )  
    hf_model_2 = AutoModelForCausalLM.from_pretrained(hf_llama2_path2)
    with torch.no_grad():
        hf_loss_2 = hf_model_2(input_ids, labels=input_ids).loss

    input_ids = input_ids.cuda()
    hf_model_1.bfloat16().cuda()
    hf_model_2.bfloat16().cuda()

    logits1 = hf_model_1(input_ids, labels=input_ids).logits.mean()
    logits2 = hf_model_2(input_ids, labels=input_ids).logits.mean()

    test_two_matrix(logits1, logits2, "HF vs. HF")
    
    # import pdb; pdb.set_trace()
    inputs = torch.randn((1, 4096, 2560)).bfloat16().cuda()
    
    outputs_1 = inputs
    outputs_2 = inputs
    with torch.no_grad():
        for i in range(32):
            outputs_1 = hf_model_1.model.layers[i](outputs_1)[0]
            outputs_2 = hf_model_2.model.layers[i](outputs_2)[0]
    x1 = outputs_1[0].view(-1)
    x2 = outputs_2[0].view(-1)
    cos_sim = F.cosine_similarity(x1, x2, dim=0)
    
    print(cos_sim)
    print(abs(x1-x2).sum())
    print(abs(x1).sum())
    print(x1.shape)
    