#!/bin/bash
# Parallel sharded MVBench eval for the motion-adaptive sampling runner.
#
# GPUs x workers/GPU shards. Deterministic stride sharding (via
# --num_shards/--shard_id) means shards never overlap; merge_mvbench.py stitches
# the per-shard files, score_mvbench.py prints the leaderboard table.
#
# Usage:
#   bash run_mvbench_parallel.sh <mode> <model_path> [output_tag] [task] [workers_per_gpu]
# Examples:
#   

#   bash run_mvbench_parallel.sh uniform  /path/to/base_models/Qwen3-VL-2B-Instruct qwen3vl-2b-mvbench-uniform  all 6
#   bash run_mvbench_parallel.sh adaptive <model> qwen3vl-2b-mvbench-adaptive action_sequence 2
#
# NOTE (OOM): adaptive mode loads Qwen3-VL + CoTracker per worker. Use 2
# workers/GPU for adaptive; uniform skips CoTracker so 6/GPU is safe.
#
# After it finishes, aggregate with:
#   python -m map_kit.benchmarks.score_mvbench --output_path predictions/<tag>
set -e

# MODE adaptive uniform
MODE="${1:?usage: run_mvbench_parallel.sh <mode> <model_path> [output_tag] [task] [workers_per_gpu] [draw_tracks] [online] [grid_size] [segment]}"
MODEL_PATH="${2:?need model_path}"
TAG="${3:-qwen3vl-2b-mvbench-${MODE}}"
TASK="${4:-all}"
WORKERS_PER_GPU="${5:-1}"
DRAW_TRACKS="${6:-on}"   # route 1: on/1 to draw motion trajectories (adaptive only)
ONLINE="${7:-off}"       # CoTracker3 streaming (lower peak memory): on/1 | off
GRID_SIZE="${8:-10}"     # CoTracker grid density (NxN); paper uses 10x10
SEGMENT="${9:-on}"       # segment_track: frame-aligned per-interval tracking (offline); paper default ON
# sanitize task for log filenames (comma-separated task lists -> safe token)
TASK_TAG="${TASK//,/-}"

ONLINE_FLAG=""
SEG_FLAG=""
case "$SEGMENT" in
  on|1|true|yes) SEG_FLAG="--segment_track" ;;
esac
# segment_track is offline-only (mutually exclusive with --online); only enable
# online when NOT segment-tracking.
if [ -z "$SEG_FLAG" ]; then
  case "$ONLINE" in
    on|1|true|yes) ONLINE_FLAG="--online" ;;
  esac
fi

# route-1 marking config is fixed to the paper setting: points selection,
# frame-to-frame trajectory span.
DRAW_FLAG=""
case "$DRAW_TRACKS" in
  on|1|true|yes) DRAW_FLAG="--draw_tracks --track_span frame --mark_mode points" ;;
esac

GPUS=(0 1 2 3)
NUM_SHARDS=$(( WORKERS_PER_GPU * ${#GPUS[@]} ))

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"
PY=python
OUT="predictions/${TAG}"

# Cap threads per worker so torch intra-op parallelism doesn't thrash CPU
# (frame decode + CoTracker are CPU-bound).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p logs "$OUT"
echo "=== MVBench | mode=$MODE | task=$TASK | model=$MODEL_PATH | $NUM_SHARDS shards -> $OUT ==="

pids=()
shard=0
for gpu in "${GPUS[@]}"; do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    CUDA_VISIBLE_DEVICES=$gpu $PY -m map_kit.benchmarks.run_mvbench \
      --mode "$MODE" --task "$TASK" \
      --model_path "$MODEL_PATH" --output_path "$OUT" \
      --tracking_fps 8 --grid_size "$GRID_SIZE" --max_new_tokens 100 \
      $DRAW_FLAG $ONLINE_FLAG $SEG_FLAG \
      --num_shards "$NUM_SHARDS" --shard_id "$shard" \
      > "logs/${TAG}.${TASK_TAG}.shard${shard}.log" 2>&1 &
    pids+=($!)
    shard=$((shard+1))
  done
done

echo "launched ${#pids[@]} shards, waiting..."
for p in "${pids[@]}"; do wait "$p"; done

$PY -m map_kit.benchmarks.merge_mvbench --output_path "$OUT" --task "$TASK"
echo "=== DONE: merged predictions at $OUT ==="
echo "=== accuracy ==="
$PY -m map_kit.benchmarks.score_mvbench --output_path "$OUT"
