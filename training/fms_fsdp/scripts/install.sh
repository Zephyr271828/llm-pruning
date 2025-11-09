conda create -n fms_fsdp python=3.10 -y
conda activate fms_fsdp

pip install torch==2.5.0 torchvision torchaudio
pip install transformers==4.46.2