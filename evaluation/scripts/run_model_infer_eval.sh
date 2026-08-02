#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVALUATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KEY="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_3DPOSE="/home/caiqingyuan/miniconda3/envs/3dpose/bin/python"
PYTHON_QWEN35="/home/caiqingyuan/data/caiqingyuan/env/lifebench-qwen35/bin/python"
EVAL_BASE="${LIFEBENCH_EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation/real_videos_single_pass}"
if [ -z "${LIFEBENCH_PREDICTIONS_ROOT:-}" ]; then
    export LIFEBENCH_PREDICTIONS_ROOT="${PROJECT_ROOT}/data/public_data_release/prediction/real_videos_single_pass"
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    printf 'Run infer and eval for model key: %s\n' "$MODEL_KEY"
    printf 'This uses one model call per video with evaluation/scripts/prompts/infer_prompt_structured.txt.\n'
    printf 'Supported model keys:\n'
    printf '  qwen25vl-7b\n  qwen35-9b\n  internvl35-8b\n  tarsier2-7b\n'
    printf '  videollama2-7b\n  videollama3-7b\n  videochat2-7b\n  video-chatgpt-7b\n'
    exit 0
fi

case "$MODEL_KEY" in
    qwen25vl-7b)
        export LIFEBENCH_QWEN_PYTHON="${LIFEBENCH_QWEN_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_DIR="${LIFEBENCH_EVAL_OUTPUT_DIR:-${EVAL_BASE}/qwen2_5vl_7b}"
        exec "${EVALUATION_DIR}/run_qwen25vl_8gpu_infer_eval.sh" "$@"
        ;;
    qwen35-9b)
        export LIFEBENCH_QWEN_PYTHON="${LIFEBENCH_QWEN_PYTHON:-${PYTHON_QWEN35}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_DIR="${LIFEBENCH_EVAL_OUTPUT_DIR:-${EVAL_BASE}/qwen3_5_9b}"
        exec "${EVALUATION_DIR}/run_qwen35_8gpu_infer_eval.sh" "$@"
        ;;
    internvl35-8b)
        export LIFEBENCH_INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-${PYTHON_QWEN35}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_DIR="${LIFEBENCH_EVAL_OUTPUT_DIR:-${EVAL_BASE}/internvl3_5_8b}"
        exec "${EVALUATION_DIR}/run_internvl35_8gpu_infer_eval.sh" "$@"
        ;;
    tarsier2-7b)
        export LIFEBENCH_INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_ROOT="${EVAL_BASE}"
        LIFEBENCH_INFERENCE_MODE=single exec "${EVALUATION_DIR}/run_tarsier2_8gpu_infer_eval.sh" "$@"
        ;;
    videollama3-7b)
        export LIFEBENCH_INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_ROOT="${EVAL_BASE}"
        LIFEBENCH_INFERENCE_MODE=single exec "${EVALUATION_DIR}/run_videollama3_8gpu_infer_eval.sh" "$@"
        ;;
    videollama2-7b|videochat2-7b|video-chatgpt-7b)
        export LIFEBENCH_INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-${PYTHON_3DPOSE}}"
        export LIFEBENCH_EVAL_OUTPUT_ROOT="${EVAL_BASE}"
        LIFEBENCH_INFERENCE_MODE=single exec "${EVALUATION_DIR}/run_legacy_model_8gpu_infer_eval.sh" \
            --model-key "$MODEL_KEY" "$@"
        ;;
    *)
        printf 'Unknown model key: %s\n' "$MODEL_KEY" >&2
        printf 'Supported keys: qwen25vl-7b qwen35-9b internvl35-8b tarsier2-7b videollama2-7b videollama3-7b videochat2-7b video-chatgpt-7b\n' >&2
        exit 2
        ;;
esac
