#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEOS_DIR="${LIFEBENCH_VIDEOS_DIR:-${PROJECT_ROOT}/data/public_data_release/real_videos}"
PREDICTIONS_ROOT="${LIFEBENCH_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/real_videos}"
PROMPTS_DIR="${LIFEBENCH_PROMPTS_DIR:-${SCRIPT_DIR}/official_evaluation/real_videos}"
NUM_GPUS="${LIFEBENCH_NUM_GPUS:-8}"

LEGACY_KEYS=(
    video-llava-7b
    videochat2-7b
    video-chatgpt-7b
    minigpt4-video
    videollama2-7b
    videollama3-7b
    mplug-owl3-7b
    tarsier2-7b
)

usage() {
    cat <<EOF
Usage: $0 [MODEL_KEY ...]

Runs inference and official real-video evaluation sequentially for the
benchmark models other than Qwen3.5-9B, Qwen2.5-VL-7B-Instruct, and
InternVL3.5-8B. With no keys, all eight models are run.

Model keys:
  ${LEGACY_KEYS[*]}

Environment overrides:
  LIFEBENCH_NUM_GPUS, LIFEBENCH_VIDEOS_DIR
  LIFEBENCH_PREDICTIONS_ROOT, LIFEBENCH_PROMPTS_DIR
  LIFEBENCH_JUDGE_PYTHON, LIFEBENCH_JUDGE_PORT_BASE
  LIFEBENCH_INFER_PYTHON, LIFEBENCH_EVAL_OUTPUT_ROOT
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "$#" -eq 0 ]; then
    REQUESTED_KEYS=("${LEGACY_KEYS[@]}")
else
    REQUESTED_KEYS=("$@")
fi

for key in "${REQUESTED_KEYS[@]}"; do
    valid=0
    for known in "${LEGACY_KEYS[@]}"; do
        if [ "$key" = "$known" ]; then
            valid=1
            break
        fi
    done
    if [ "$valid" -ne 1 ]; then
        echo "Unknown legacy model key: $key" >&2
        usage >&2
        exit 2
    fi
done

for key in "${REQUESTED_KEYS[@]}"; do
    echo "============================================================"
    echo "Running LifeBench infer + eval for ${key}"
    echo "============================================================"
    "$SCRIPT_DIR/run_legacy_model_8gpu_infer_eval.sh" \
        --model-key "$key" \
        --videos-dir "$VIDEOS_DIR" \
        --predictions-root "$PREDICTIONS_ROOT" \
        --prompts-dir "$PROMPTS_DIR" \
        --num-gpus "$NUM_GPUS"
done
