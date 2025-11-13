#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n llm-pruner python=3.9 -y
conda activate llm-pruner

pip install -r requirement.txt