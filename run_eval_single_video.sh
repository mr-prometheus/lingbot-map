#!/bin/bash
#SBATCH -J lingbot_eval_single
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=runs/eval_single_%j.out

# ==== CONFIGURATION ====
VIDEO_DIR="/home/c3-0/datasets/BDD_Dataset/Videos/bdd100k/videos/train/"
FILENAME_LIST="gama_list/train_day.list"
TEMP_FRAMES_DIR="extracted_clips_train_tmp"
OUTPUT_DIR="/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/vggt-output-train-rendered-lingbot"
MODEL_PATH="/home/de575594/Deepan/CV/geolocalization/lingbot-map/checkpoints/lingbot-map-long.pt"
PROGRESS_CSV="runs/eval_progress_lingbot.csv"
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

mkdir -p runs "$OUTPUT_DIR" "$TEMP_FRAMES_DIR"

FORCE_FLAG=""
if [ "$FORCE_RECOMPUTE" = true ]; then
    FORCE_FLAG="--force"
fi

# ── Pick the first video from the list ──────────────────────────────────────
video_stem=$(head -1 "$FILENAME_LIST" | awk '{print $1}')
video_file="${VIDEO_DIR}/${video_stem}.mov"

echo "===================================================="
echo "LingBot-Map Inference + Render Pipeline (EVAL)"
echo "Video stem   : $video_stem"
echo "Video file   : $video_file"
echo "Temp frames  : $TEMP_FRAMES_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Model        : $MODEL_PATH"
echo "Progress CSV : $PROGRESS_CSV"
echo "Conf thresh  : $CONF_THRESHOLD"
echo "Scale frames : $NUM_SCALE_FRAMES"
echo "Force        : $FORCE_RECOMPUTE"
echo "Start time   : $(date)"
echo "===================================================="

if [ ! -f "$video_file" ]; then
    echo "[ERROR] Video not found: $video_file"
    exit 1
fi

# ── Step 1: Extract clips from the video ────────────────────────────────────
echo ""
echo "[1/3] Extracting clips from $video_stem ..."

export VGGT_VIDEO_FILE="$video_file"
export VGGT_VIDEO_STEM="$video_stem"
export VGGT_TEMP_FRAMES="$TEMP_FRAMES_DIR"

python - <<'PYEOF'
import os, cv2
from pathlib import Path

video_file  = Path(os.environ["VGGT_VIDEO_FILE"])
video_stem  = os.environ["VGGT_VIDEO_STEM"]
temp_frames = Path(os.environ["VGGT_TEMP_FRAMES"])

CLIP_DURATION_SEC = 1.0
FRAMES_PER_CLIP   = 8

cap = cv2.VideoCapture(str(video_file))
if not cap.isOpened():
    print(f"[ERROR] Cannot open {video_file}")
    raise SystemExit(1)

src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
clip_frames = int(round(src_fps * CLIP_DURATION_SEC))   # frames that span 1 s
step        = max(1, clip_frames // FRAMES_PER_CLIP)    # sample every N frames

print(f"  FPS={src_fps:.1f}  total={total_fr}  clip_span={clip_frames}  step={step}")

out_base = temp_frames / video_stem
clip_idx = 0
frame_idx = 0

while True:
    clip_dir = out_base / f"clip_{clip_idx:04d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for _ in range(FRAMES_PER_CLIP):
        # seek to the right position in the clip
        target = clip_idx * clip_frames + saved * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(clip_dir / f"frame_{saved:04d}.jpg"), frame)
        saved += 1

    if saved == 0:
        clip_dir.rmdir()
        break

    clip_idx += 1
    # stop if we've passed the end of the video
    if clip_idx * clip_frames >= total_fr:
        break

cap.release()
print(f"  Extracted {clip_idx} clips -> {out_base}")
PYEOF

extract_status=$?
if [ $extract_status -ne 0 ]; then
    echo "[ERROR] Frame extraction failed (exit $extract_status)"
    exit 1
fi

clips_extracted=$(ls -d "$TEMP_FRAMES_DIR/$video_stem"/clip_* 2>/dev/null | wc -l)
echo "  Clips extracted: $clips_extracted"

if [ "$clips_extracted" -eq 0 ]; then
    echo "[ERROR] No clips extracted for $video_stem"
    exit 1
fi

# ── Step 2: Run LingBot-Map inference ───────────────────────────────────────
echo ""
echo "[2/3] Running LingBot-Map inference on $clips_extracted clips..."

python inference_render_eval.py \
    "$TEMP_FRAMES_DIR/$video_stem" \
    "$OUTPUT_DIR" \
    --model_path       "$MODEL_PATH" \
    --progress_csv     "$PROGRESS_CSV" \
    --conf_threshold   "$CONF_THRESHOLD" \
    --num_scale_frames "$NUM_SCALE_FRAMES" \
    $FORCE_FLAG

infer_status=$?

# ── Step 3: Clean up temp frames ────────────────────────────────────────────
echo ""
echo "[3/3] Cleaning up temp frames for $video_stem..."
rm -rf "${TEMP_FRAMES_DIR:?}/$video_stem"
rmdir "$TEMP_FRAMES_DIR" 2>/dev/null   # only removes if empty

echo ""
echo "===================================================="
if [ $infer_status -ne 0 ]; then
    echo "Pipeline FAILED (exit $infer_status)"
else
    echo "Pipeline complete!"
    echo "Results in : $OUTPUT_DIR/$video_stem"
fi
echo "End time : $(date)"
echo "===================================================="

exit $infer_status
