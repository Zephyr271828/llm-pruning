#!/bin/bash

conda create -n minitron python=3.9 -y 
conda activate minitron

pip install torch==2.5.0 torchvision torchaudio transformers==4.46.2 lm-eval==0.4.9 matplotlib