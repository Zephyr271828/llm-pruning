#!/bin/bash

conda create -n wanda python=3.9 -y
conda activate wanda

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
pip install transformers==4.43.3 datasets==2.16.0 wandb sentencepiece accelerate==0.31.0

# conda create -n wanda-eval python=3.9 -y
# conda activate wanda-eval
# cd lm-evaluation-harness 
# pip install -e .