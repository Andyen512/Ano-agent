#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEOS_DIR="${LIFEBENCH_VIDEOS_DIR:-${PROJECT_ROOT}/data/public_data_release/real_videos}"
PREDICTIONS_ROOT="${LIFEBENCH_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/real_videos_gt_hint}"
GT_HINT_DIR="${LIFEBENCH_GT_HINT_DIR:-${PROJECT_ROOT}/data/public_data_release/annotations/real_videos}"
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

GT-hint experiment for the eight legacy benchmark models:
  - Only abnormal / risk_only videos (GT risk_status, must have GT time_spans).
  - The GT risk interval is injected into the perception / cognition /
    planning prompts as the nearest sampled-frame ids (Frame X to Frame Y,
    sampled frames map back to seconds via the per-video frame time mapping).
  - The temporal (grounding) dimension is skipped in inference.
  - Evaluation only covers abnormal / risk_only and excludes temporal metrics
    (overall = perception/cognition/intervention renormalized).

With no keys, all eight models are run.

Model keys:
  ${LEGACY_KEYS[*]}

Environment overrides:
  LIFEBENCH_NUM_GPUS, LIFEBENCH_VIDEOS_DIR
  LIFEBENCH_PREDICTIONS_ROOT, LIFEBENCH_PROMPTS_DIR
  LIFEBENCH_GT_HINT_DIR, LIFEBENCH_JUDGE_PYTHON
  LIFEBENCH_JUDGE_PORT_BASE, LIFEBENCH_INFER_PYTHON
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

export LIFEBENCH_INFERENCE_MODE=gt_hint
export LIFEBENCH_EVAL_SKIP_TEMPORAL=1
export LIFEBENCH_EVAL_RISK_STATUS_FILTER=abnormal,risk_only

for key in "${REQUESTED_KEYS[@]}"; do
    echo "============================================================"
    echo "Running LifeBench GT-hint infer + eval for ${key}"
    echo "============================================================"
    "$SCRIPT_DIR/run_legacy_model_8gpu_infer_eval.sh" \
        --model-key "$key" \
        --videos-dir "$VIDEOS_DIR" \
        --predictions-root "$PREDICTIONS_ROOT" \
        --prompts-dir "$PROMPTS_DIR" \
        --num-gpus "$NUM_GPUS"
done
