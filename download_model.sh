#!/bin/bash
#SBATCH -J lingbot_download
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=runs/download_%j.out

# ==== CONFIGURATION ====
MODEL_DIR="/home/de575594/Deepan/CV/geolocalization/lingbot-map/checkpoints"
DOWNLOAD_LONG=true      # lingbot-map-long.pt  (recommended, 4.63 GB)
DOWNLOAD_BASE=false     # lingbot-map.pt        (balanced, ~4.6 GB)
DOWNLOAD_STAGE1=false   # lingbot-map-stage1.pt (stage-1, 4.76 GB)
# =======================


export MODEL_DIR DOWNLOAD_LONG DOWNLOAD_BASE DOWNLOAD_STAGE1

module purge
module load anaconda3
module load cuda/12.8

export PATH="/home/de575594/.conda/envs/lingbot-map/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate lingbot-map

mkdir -p runs "$MODEL_DIR"

echo "===================================================="
echo "Downloading LingBot-Map checkpoints"
echo "Destination : $MODEL_DIR"
echo "Start time  : $(date)"
echo "===================================================="

python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "robbyant/lingbot-map"
dest    = Path(os.environ["MODEL_DIR"])
token   = os.environ.get("HF_TOKEN")

files = []
if os.environ.get("DOWNLOAD_LONG")   == "true": files.append("lingbot-map-long.pt")
if os.environ.get("DOWNLOAD_BASE")   == "true": files.append("lingbot-map.pt")
if os.environ.get("DOWNLOAD_STAGE1") == "true": files.append("lingbot-map-stage1.pt")

for fname in files:
    out_path = dest / fname
    if out_path.exists():
        print(f"[SKIP] {fname} already exists at {out_path}")
        continue
    print(f"[DOWNLOAD] {fname} ...")
    hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(dest), token=token)
    print(f"[OK] {fname} -> {out_path}")

print("Done.")
PYEOF

echo ""
echo "===================================================="
echo "Download complete: $(date)"
ls -lh "$MODEL_DIR"
echo "===================================================="
