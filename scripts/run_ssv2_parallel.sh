#!/bin/bash
# Parallel sharded SSv2 eval for the motion-adaptive sampling runner (route 2 +
# optional route 1 marks). Mirrors run_mvbench_parallel.sh but targets the SSv2
# 4-way MC subset via map_kit/benchmarks/run_ssv2.py.
#
# Deterministic stride sharding (--num_shards/--shard_id); merge_mvbench.py
# stitches per-shard files, score_mvbench.py prints the leaderboard.
#
# Usage:
#   bash run_ssv2_parallel.sh <mode> <model_path> [output_tag] [workers_per_gpu] \
#        [draw_tracks] [online] [grid_size] [segment]
# Examples:
#   bash run_ssv2_parallel.sh uniform  <model> qwen3vl-2b-ssv2-uniform "" 4 off
#   bash run_ssv2_parallel.sh adaptive <model> qwen3vl-2b-ssv2-adaptive-marks "" 2 on off 20 on
#
# NOTE (OOM): adaptive loads Qwen3-VL + CoTracker per worker -> 2 workers/GPU.
#   uniform (no CoTracker unless --draw_tracks) can go higher.
# route-1 marks use points selection + frame-aligned two-segment tracking
#   (segment on, which is offline; mutually exclusive with online).
set -e

MODE="${1:?usage: run_ssv2_parallel.sh <mode> <model_path> [tag] [workers_per_gpu] [draw_tracks] [online] [grid_size] [segment]}"
MODEL_PATH="${2:?need model_path}"
TAG="${3:-qwen3vl-2b-ssv2-${MODE}}"
WORKERS_PER_GPU="${4:-2}"
DRAW_TRACKS="${5:-off}"    # route 1: on/1 to draw motion trajectories (adaptive only)
ONLINE="${6:-off}"         # CoTracker3 streaming (lower peak mem): on/1 | off
GRID_SIZE="${7:-10}"       # CoTracker grid density (NxN); paper uses 10x10
SEGMENT="${8:-on}"        # segment_track: frame-aligned per-interval tracking (offline); paper default ON

ONLINE_FLAG=""
SEG_FLAG=""
case "$SEGMENT" in
  on|1|true|yes) SEG_FLAG="--segment_track" ;;
esac
# segment_track is offline-only (mutually exclusive with --online).
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

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

mkdir -p logs "$OUT"
echo "=== SSv2 | mode=$MODE | model=$MODEL_PATH | draw=$DRAW_TRACKS | $NUM_SHARDS shards -> $OUT ==="

pids=()
shard=0
for gpu in "${GPUS[@]}"; do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    CUDA_VISIBLE_DEVICES=$gpu $PY -m map_kit.benchmarks.run_ssv2 \
      --mode "$MODE" \
      --model_path "$MODEL_PATH" --output_path "$OUT" \
      --tracking_fps 8 --grid_size "$GRID_SIZE" --max_new_tokens 8192 \
      $DRAW_FLAG $ONLINE_FLAG $SEG_FLAG \
      --num_shards "$NUM_SHARDS" --shard_id "$shard" \
      > "logs/${TAG}.shard${shard}.log" 2>&1 &
    pids+=($!)
    shard=$((shard+1))
  done
done

echo "launched ${#pids[@]} shards, waiting..."
for p in "${pids[@]}"; do wait "$p"; done

$PY -m map_kit.benchmarks.merge_mvbench --output_path "$OUT" --task all
echo "=== DONE: merged predictions at $OUT ==="
echo "=== accuracy ==="
$PY -m map_kit.benchmarks.score_mvbench --output_path "$OUT"
