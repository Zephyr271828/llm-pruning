#!/bin/bash

conda create -n fms_fsdp python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms_fsdp

pip install torch==2.5.0 torchvision torchaudio accelerate
pip install transformers==4.43.3 datasets==4.4.1 peft==0.13.0
pip install fire pyarrow torchdata wandb hydra-core sacrebleu 