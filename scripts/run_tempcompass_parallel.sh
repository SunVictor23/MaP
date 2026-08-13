#!/bin/bash
# Parallel sharded TempCompass eval for the motion-adaptive sampling runner.
#
# 4 GPUs x 4 workers/GPU = 16 shards. Deterministic stride sharding (via
# --num_shards/--shard_id) means shards never overlap; merge_shards.py stitches
# the per-shard files into the final <task>.json for scoring.
#
# Usage:
#   bash run_yes_no_parallel.sh <mode> <model_path> [output_tag]
# Examples:
#   bash run_yes_no_parallel.sh adaptive /path/to/base_models/Qwen3-VL-2B-Instruct qwen3vl-2b-adaptive
#   bash run_yes_no_parallel.sh uniform  /path/to/base_models/Qwen3-VL-8B-Instruct qwen3vl-8b-uniform
#
# After it finishes, score with:
#   cd /path/to/TempCompass
#   ln -sfn <MotionVLM>/predictions/<tag> predictions/<tag>
#   python eval_yes_no.py --video_llm <tag> --disable_llm
set -e

MODE="${1:?usage: run_tempcompass_parallel.sh <mode> <model_path> [tag] [task_type] [workers_per_gpu]}"
MODEL_PATH="${2:?need model_path}"
TAG="${3:-qwen3vl-${MODE}}"
TASK="${4:-caption_matching}"
# adaptive runs CoTracker per worker -> keep ~2/GPU (OOM); uniform can use 6.
WORKERS_PER_GPU="${5:-2}"
ONLINE="${6:-off}"   # CoTracker3 streaming (lower peak mem, OOM-safe): on/off
SEGMENT="${7:-on}"  # segment_track: frame-aligned per-interval tracking (offline); paper default ON
GRID_SIZE="${8:-10}" # CoTracker grid density (NxN); paper uses 10x10
GPUS=(0 1 2 3)
NUM_SHARDS=$(( WORKERS_PER_GPU * ${#GPUS[@]} ))

ONLINE_FLAG=""
SEG_FLAG=""
case "$SEGMENT" in on|1|true|yes) SEG_FLAG="--segment_track" ;; esac
# segment_track is offline-only; only enable online when NOT segment-tracking.
if [ -z "$SEG_FLAG" ]; then
  case "$ONLINE" in on|1|true|yes) ONLINE_FLAG="--online" ;; esac
fi

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"
PY=python
OUT="predictions/${TAG}"
OUT_ABS="${PKG_DIR}/${OUT}"

# Cap threads per worker: 16 torch processes on 184 cores would each grab ~184
# intra-op threads and thrash CPU, starving the GPUs (frame decode + CoTracker
# are CPU-bound). 8 threads/worker keeps it bounded.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p logs "$OUT"
echo "=== $TASK | mode=$MODE | model=$MODEL_PATH | $NUM_SHARDS shards -> $OUT ==="

pids=()
shard=0
for gpu in "${GPUS[@]}"; do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    CUDA_VISIBLE_DEVICES=$gpu $PY -m map_kit.benchmarks.run_tempcompass \
      --mode "$MODE" --task_type "$TASK" \
      --model_path "$MODEL_PATH" --output_path "$OUT" \
      --tracking_fps 8 --grid_size "$GRID_SIZE" --max_new_tokens 64 \
      $ONLINE_FLAG $SEG_FLAG \
      --num_shards "$NUM_SHARDS" --shard_id "$shard" \
      > "logs/${TAG}.${TASK}.shard${shard}.log" 2>&1 &
    pids+=($!)
    shard=$((shard+1))
  done
done

echo "launched ${#pids[@]} shards, waiting..."
for p in "${pids[@]}"; do wait "$p"; done

$PY /path/to/TempCompass/merge_shards.py \
  --output_path "$OUT" --task_type "$TASK"
echo "=== DONE: merged predictions at $OUT/${TASK}.json ==="

# --- score inline (rule match, no LLM) ---
TC_DIR=/path/to/TempCompass
SCORER="eval_${TASK}.py"
if [ -f "$TC_DIR/$SCORER" ]; then
  echo "=== accuracy ($TASK, --disable_llm) ==="
  ( cd "$TC_DIR" \
    && ln -sfn "$OUT_ABS" "predictions/${TAG}" \
    && $PY "$SCORER" --video_llm "$TAG" --disable_llm )
else
  echo "[skip scoring] no scorer $TC_DIR/$SCORER"
fi
