#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_GPUS="${LIFEBENCH_NUM_GPUS:-8}"

exec "$SCRIPT_DIR/run_legacy_model_8gpu_infer_eval.sh" \
    --model-key videollama3-7b \
    --num-gpus "$NUM_GPUS" \
    "$@"
