import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
import json    
import argparse
import matplotlib.pyplot as plt 
from math import log

def plot_scores(data, save_path=None, take_log=False):
    if take_log:
        data["scores"] = [log(score) if score > 0 else 1e-10 for score in data["scores"]]
        data["base_line"] = log(data["base_line"]) if data["base_line"] > 0 else 1e-10
    
    plt.plot(data["scores"], marker='o', label='Pruned Model Scores')
    plt.axhline(y=data["base_line"], color='r', linestyle='--', label='Baseline Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        
def plot_bi_score(data, save_path=None):
        
    model_name = data.get("model_name", "Model")
    
    plt.plot(data["bi_scores"], marker='o', label='Pruned Model Scores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.log_path, 'r') as f:
        data = json.load(f)
        
    save_path = args.log_path.replace('.json', '.png')
    if "base_line" in data and "scores" in data:
        take_log = any(task in args.log_path for task in ['wikitext', 'c4', 'cnn_dailymail', 'pg19'])
        plot_scores(data, save_path, take_log=take_log)
    elif "bi_scores" in data:
        plot_bi_score(data, save_path)  
    print(f"Plot saved to {save_path}")