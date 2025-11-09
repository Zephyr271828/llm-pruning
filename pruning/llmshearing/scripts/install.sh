#!/bin/bash

conda create -n llmshearing python=3.9 -y 
conda activate llmshearing

pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "flash-attn==2.3.2"     

pip install -r llmshearing/requirements.txt

cd llmshearing
pip install -e .