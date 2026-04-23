#!/bin/bash
#SBATCH -J lingbot_map_setup
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem12
#SBATCH --time=02:00:00
#SBATCH --output=runs/setup_job.out

module load anaconda3
module load cuda/12.8

export PATH="/home/de575594/.conda/envs/lingbot-map/bin:$PATH"
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map

# PyTorch with CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install lingbot-map (editable)
pip install -e .

# FlashInfer for paged KV cache attention (CUDA 12.8 + PyTorch 2.9)
pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.9/

# Optional: sky masking (GPU-accelerated)
pip install onnxruntime-gpu
