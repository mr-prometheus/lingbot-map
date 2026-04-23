#!/bin/bash
#SBATCH -J lingbot_infer
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/infer_%j.out

# ==== CONFIGURATION ====
VIDEO_PATH="videos/venice.mp4"  # <-- set this to your video path
OUTPUT_DIR="results"
MODEL_PATH="/home/de575594/Deepan/CV/geolocalization/lingbot-map/checkpoints/lingbot-map-long.pt"
FPS=10
CONF_THRESHOLD=1.5
NUM_SCALE_FRAMES=8
# =======================

export PATH="/home/de575594/.conda/envs/lingbot-map/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/12.4 2>/dev/null || module load cuda/12.1 2>/dev/null || true

conda activate lingbot-map

if [ -z "$VIDEO_PATH" ]; then
    echo "[ERROR] Set VIDEO_PATH in the script before submitting."
    exit 1
fi

mkdir -p runs "$OUTPUT_DIR"

echo "===================================================="
echo "LingBot-Map Simple Inference"
echo "Video      : $VIDEO_PATH"
echo "Output     : $OUTPUT_DIR"
echo "Model      : $MODEL_PATH"
echo "FPS        : $FPS"
echo "Conf thresh: $CONF_THRESHOLD"
echo "Start time : $(date)"
echo "===================================================="

python infer.py \
    --video_path       "$VIDEO_PATH" \
    --model_path       "$MODEL_PATH" \
    --output_dir       "$OUTPUT_DIR" \
    --fps              "$FPS" \
    --conf_threshold   "$CONF_THRESHOLD" \
    --num_scale_frames "$NUM_SCALE_FRAMES"

status=$?

echo "===================================================="
if [ $status -ne 0 ]; then
    echo "FAILED (exit $status)"
else
    video_id=$(basename "${VIDEO_PATH%.*}")
    echo "Done! Results in: $OUTPUT_DIR/$video_id/"
fi
echo "End time : $(date)"
echo "===================================================="

exit $status
