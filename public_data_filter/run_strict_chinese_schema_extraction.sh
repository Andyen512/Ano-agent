#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data_4/liuyuan/lifebench"
SCRIPT="$PROJECT_ROOT/public_data_filter/extract_strict_chinese_schema.py"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    /data_4/liuyuan/anaconda3/envs/swift-3.6/bin/python \
    /data_4/liuyuan/anaconda3/envs/lifebench-vlm/bin/python \
    /data_4/liuyuan/anaconda3/envs/lifebench/bin/python \
    python
  do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import torch
import transformers
print("ok")
PY
    then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find a Python environment with torch and transformers." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

REVIEWS_ROOT="$PROJECT_ROOT/public_data_filter/batch_outputs/persistent_full/reviews"
OUTPUT_ROOT="$PROJECT_ROOT/public_data_filter/batch_outputs/persistent_full"
OUTPUT_STEM="strict_english_schema"
MODEL_PATH="$PROJECT_ROOT/models/Qwen__Qwen3-8B-Base"
COMMON_ARGS=(
  --reviews-root "$REVIEWS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --output-stem "$OUTPUT_STEM"
  --model-path "$MODEL_PATH"
  --wait-for-model
  --resume
  "$@"
)

MANUAL_SHARD_ARGS=0
for arg in "$@"; do
  case "$arg" in
    --num-shards|--num-shards=*|--shard-index|--shard-index=*|--merge-shards)
      MANUAL_SHARD_ARGS=1
      ;;
  esac
done

GPU_LIST_RAW="${SCHEMA_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
IFS=',' read -r -a RAW_GPU_IDS <<< "$GPU_LIST_RAW"
GPU_IDS=()
for gpu in "${RAW_GPU_IDS[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  if [[ -n "$gpu" ]]; then
    GPU_IDS+=("$gpu")
  fi
done

if [[ "$MANUAL_SHARD_ARGS" -eq 1 || "${#GPU_IDS[@]}" -le 1 ]]; then
  exec "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}"
fi

LOG_DIR="$OUTPUT_ROOT/parallel_logs"
mkdir -p "$LOG_DIR"
export TOKENIZERS_PARALLELISM=false
POLL_SECONDS="${SCHEMA_PROGRESS_POLL_SECONDS:-30}"
NUM_SHARDS="${#GPU_IDS[@]}"
PIDS=()

echo "[launcher] starting $NUM_SHARDS shard workers for $OUTPUT_STEM"
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  shard_label="$(printf 'shard%02dof%02d' "$idx" "$NUM_SHARDS")"
  shard_log="$LOG_DIR/${OUTPUT_STEM}.${shard_label}.log"
  echo "[launcher] shard $idx/$((NUM_SHARDS - 1)) uses GPU $gpu -> $shard_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$idx" \
      --device-map cuda:0
  ) >"$shard_log" 2>&1 &
  PIDS+=("$!")
done

LAST_DONE=-1
while :; do
  alive=0
  done_rows=0
  for idx in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$idx]}" 2>/dev/null; then
      alive=1
    fi
    shard_jsonl="$OUTPUT_ROOT/${OUTPUT_STEM}.shard$(printf '%02d' "$idx")of$(printf '%02d' "$NUM_SHARDS").jsonl"
    if [[ -f "$shard_jsonl" ]]; then
      count="$(wc -l < "$shard_jsonl")"
      done_rows=$((done_rows + count))
    fi
  done
  if [[ "$done_rows" -ne "$LAST_DONE" ]]; then
    echo "[launcher] completed rows: $done_rows"
    LAST_DONE="$done_rows"
  fi
  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

status=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "[launcher] shard $idx failed; inspect $LOG_DIR/${OUTPUT_STEM}.shard$(printf '%02d' "$idx")of$(printf '%02d' "$NUM_SHARDS").log" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[launcher] one or more shards failed; rerun the same command to resume." >&2
  exit "$status"
fi

MERGE_LOG="$LOG_DIR/${OUTPUT_STEM}.merge.log"
echo "[launcher] merging shard outputs -> $MERGE_LOG"
"$PYTHON_BIN" "$SCRIPT" \
  --reviews-root "$REVIEWS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --output-stem "$OUTPUT_STEM" \
  --model-path "$MODEL_PATH" \
  --num-shards "$NUM_SHARDS" \
  --merge-shards >"$MERGE_LOG" 2>&1
cat "$MERGE_LOG"
