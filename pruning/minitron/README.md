# Minitron \& ShortGPT

This directory contains my implementation of Minitron{depth, width} and ShortGPT. The key reasons we're doing this are:
1. Although NVIDIA provides an official impplementation for network pruning, the structure of the code is quite complex and non-trivial to hack for research use.
2. ShortGPT does not provide an official implementation. The best resource we could find was [an unofficial implementation](https://github.com/sramshetty/ShortGPT) by Shivaen Ramshetty.

## Installation
```bash
bash scripts/install.sh
```

## Minitron
**Pruning**

Run the following command to replicate the results in the minitron paper.
```bash
bash scripts/prune_llama3.1-8b.sh

python plot_scores.py 
```

**Results**

Winogrande

Paper:
![](figs/minitron-winogrande.png)

Ours:
![](figs/minitron-winogrande-ours.png)

Wikitext:

Paper:
![](figs/minitron-wikitext.png)

Ours:
![](figs/minitron-wikitext-ours.png)

## ShortGPT
**Pruning**

Run the following command to replicate the results in the shortgpt paper.
```bash
bash scripts/prune_llama2-7b.sh
```

**Results**

Paper:
![](figs/shortgpt-fig3.png)

Ours:
![](figs/shortgpt-fig3-ours.png)

## References
1. [minitron paper 1](https://arxiv.org/abs/2407.14679) | [minitron paper 2](https://arxiv.org/abs/2408.11796)
2. [nemotron](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/qwen/pruning-distillation) | [modelopt-tensorRT](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/3_pruning.html)
3. [shortgpt paper](https://arxiv.org/abs/2403.03853)
4. [unofficial implementation of shortgpt](https://arxiv.org/abs/2403.03853)
