# Wanda, SparseGPT, and Magnitude

This directory contains the implementation of Wanda, SparseGPT, and Magnitude pruning for LLMs by Sun et al.

## Installation
```bash
bash scripts/install.sh
```
Note: we are NOT exactly following the original installation guide since older versions of transformers are incompatible with newer models (e.g. Llama-3.1-8B). For original installation guide, see [here](src/INSTALL.md).

## Pruning
```bash
bash scripts/prune_llama3.1-8b.sh
bash scripts/prune_llama2-7b.sh
bash scripts/prune_llama-7b.sh
```

## Results

**Wanda**

Llama-2-7b-hf:
| Sparsity | Ratio | Source | BoolQ | RTE | Hellaswag | Winogrande | ARC-E | ARC-C | OBQA |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| unstructured | 0.5 | Paper | 75.0 | 53.4 | 52.5 | 68.2 | 72.8 | 39.9 | 31.2 |
| unstructured | 0.5 | Ours  | 76.7 | 53.4 | 52.5 | 68.7 | 72.4 | 39.4 | 30.8 |
| 4:8          | 0.5 | Paper | 72.7 | 53.8 | 46.5	| 66.6	| 66.7 | 34.1	| 25.8 |
| 4:8          | 0.5 | Ours  | 73.0 | 53.8 | 46.9 | 66.9 | 67.0 | 34.0 | 26.2 |
| 2:4          | 0.5 | Paper | 67.7 | 53.0 | 40.9	| 62.4 |	61.78 | 31.2 | 24.2 |
| 2:4          | 0.5 | Ours  | 68.0 | 53.4 | 41.2 | 62.6 | 62.6 | 30.9 | 23.8 |

**SparseGPR**

Llama-2-7b-hf:
| Sparsity | Ratio | Source | BoolQ | RTE | Hellaswag | Winogrande | ARC-E | ARC-C | OBQA |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| unstructured | 0.5 | Paper | 75.0 | 54.2 | 52.4 | 69.9 | 73.3 | 39.9 | 29.2 |
| unstructured | 0.5 | Ours  | 73.7 | 53.8 | 52.8 | 70.0 | 72.0 | 38.5 | 29.2 |
| 4:8          | 0.5 | Paper | 72.7 | 55.2 | 48.2	| 68.1	| 69.2 | 35.8	| 27.4 |
| 4:8          | 0.5 | Ours  | 72.5 | 56.7 | 48.2 | 67.3 | 69.0 | 35.2 | 27.6 |
| 2:4          | 0.5 | Paper | 70.5 | 58.8 |	43.3 | 66.7 |	64.1 | 30.0	| 23.2	|
| 2:4          | 0.5 | Ours  | 70.3 | 58.5 | 43.3 | 64.7 | 64.0 | 31.6 | 24 |

**Magnitude**

Llama-2-7b-hf:
| Sparsity | Ratio | Source | BoolQ | RTE | Hellaswag | Winogrande | ARC-E | ARC-C | OBQA |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| unstructured | 0.5 | Paper | 63.0 | 57.0 |	49.1 | 63.3 | 64.1	| 34.6	| 26.8 |
| unstructured | 0.5 | Ours  | 62.9 | 57.0 | 49.1 | 63.2 | 64.1 | 34.6 | 26.8 |
| 4:8          | 0.5 | Paper | 63.0 | 52.4 | 50.1 | 62.4 | 64.7	| 35.9 | 26.0 | 
| 4:8          | 0.5 | Ours  | 63.0 | 52.4 | 50.1 | 62.4 | 64.8 | 35.9 | 26.0 |
| 2:4          | 0.5 | Paper | 56.2 | 51.4 |	42.3 |		60.9 | 59.2 | 27.3 | 21.8 |
| 2:4          | 0.5 | Ours  | 59.8 | 52.4 | 45.4 | 61.1 | 61.9 | 30.2 | 21.8 | 

## References
1. [wanda paper](https://arxiv.org/abs/2306.11695)
2. [wanda repo](https://github.com/locuslab/wanda)
3. [sparsegpt paper](https://arxiv.org/abs/2301.00774)
4. [sparsegpt codebase](https://github.com/IST-DASLab/sparsegpt)
5. [magnitude pruning paper](https://arxiv.org/abs/1506.02626)




