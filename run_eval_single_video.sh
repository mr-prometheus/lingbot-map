#!/bin/bash
#SBATCH -J lingbot_eval_single
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/eval_single_%j.out

# ==== CONFIGURATION ====
VIDEO_DIR="/path/to/extracted_clips/video_stem"   # dir with clip_* subdirs
OUTPUT_DIR="/path/to/output/eval"
MODEL_PATH="/home/de575594/Deepan/CV/geolocalization/lingbot-map/checkpoints/lingbot-map-long.pt"
PROGRESS_CSV="runs/eval_progress.csv"
CONF_THRESHOLD=0.5
NUM_SCALE_FRAMES=8
FORCE_RECOMPUTE=false
# =======================

export PATH="/home/de575594/.conda/envs/lingbot-map/bin:$PATH"
eval "$(conda shell.bash hook)"
module purge
module load anaconda3
module load cuda/12.8

conda activate lingbot-map

mkdir -p runs "$OUTPUT_DIR"

FORCE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
    FORCE_FLAG="--force"
fi

echo "===================================================="
echo "LingBot-Map Inference + Render Pipeline (EVAL)"
echo "Video dir    : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Model        : $MODEL_PATH"
echo "Progress CSV : $PROGRESS_CSV"
echo "Conf thresh  : $CONF_THRESHOLD"
echo "Scale frames : $NUM_SCALE_FRAMES"
echo "Force        : $FORCE_RECOMPUTE"
echo "Start time   : $(date)"
echo "===================================================="

python inference_render_eval.py \
    "$VIDEO_DIR" \
    "$OUTPUT_DIR" \
    --model_path       "$MODEL_PATH" \
    --progress_csv     "$PROGRESS_CSV" \
    --conf_threshold   "$CONF_THRESHOLD" \
    --num_scale_frames "$NUM_SCALE_FRAMES" \
    $FORCE_FLAG

exit_code=$?

echo ""
echo "===================================================="
echo "Exit code : $exit_code"
echo "End time  : $(date)"
echo "===================================================="

exit $exit_code
