#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEOS_DIR="${LIFEBENCH_VIDEOS_DIR:-${PROJECT_ROOT}/data/public_data_release/real_videos}"
PREDICTIONS_ROOT="${LIFEBENCH_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/real_videos}"
GT_DIR="${LIFEBENCH_GT_DIR:-${PROJECT_ROOT}/data/public_data_release/annotations/real_videos}"
GT_HINT_DIR="${LIFEBENCH_GT_HINT_DIR:-$GT_DIR}"
PROMPTS_DIR="${LIFEBENCH_PROMPTS_DIR:-${SCRIPT_DIR}/official_evaluation/real_videos}"
EVAL_OUTPUT_ROOT="${LIFEBENCH_EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation/real_videos}"
# Keep generated compatibility directories and temporary model files off the
# nearly-full root filesystem.
TMP_ROOT="${LIFEBENCH_TMPDIR:-${HOME}/data/caiqingyuan/env}"
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_ROOT}/pip-cache}"
LEGACY_SITE="${LIFEBENCH_LEGACY_SITE:-${TMP_ROOT}/legacy_site}"
if [ -d "$LEGACY_SITE" ]; then
    export PYTHONPATH="$LEGACY_SITE${PYTHONPATH:+:${PYTHONPATH}}"
fi
# The legacy model stack needs CUDA 11.8-compatible PyTorch. Keep it separate
# from the newer Qwen3.5/InternVL Transformers runtime.
INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-${LIFEBENCH_LEGACY_PYTHON:-/home/caiqingyuan/miniconda3/envs/3dpose/bin/python}}"
JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-/home/caiqingyuan/miniconda3/envs/3dpose/bin/python}"
NUM_GPUS="${LIFEBENCH_NUM_GPUS:-8}"
JUDGE_PORT_BASE="${LIFEBENCH_JUDGE_PORT_BASE:-18180}"
CPU_THREADS="${LIFEBENCH_CPU_THREADS:-2}"
INFERENCE_MODE="${LIFEBENCH_INFERENCE_MODE:-sequential}"
MODEL_KEY=""

LOG_DIR="$(readlink -f "${PROJECT_ROOT}/logs")"
JUDGE_SCRIPT="${SCRIPT_DIR}/official_evaluation/real_videos/judge_server.py"
EVAL_SCRIPT="${SCRIPT_DIR}/official_evaluation/real_videos/evaluate.py"
JUDGE_MODEL_PATH="${LIFEBENCH_JUDGE_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen__Qwen3-8B}"

declare -A MODEL_IDS=(
    [video-llava-7b]="LanguageBind/Video-LLaVA-7B"
    [videochat2-7b]="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf"
    [video-chatgpt-7b]="MBZUAI/Video-ChatGPT-7B"
    [minigpt4-video]="Vision-CAIR/MiniGPT4-Video"
    [videollama2-7b]="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    [videollama3-7b]="DAMO-NLP-SG/VideoLLaMA3-7B"
    [mplug-owl3-7b]="mPLUG/mPLUG-Owl3-7B-241101"
    [tarsier2-7b]="omni-research/Tarsier2-7b-0115"
)

usage() {
    cat <<EOF
Usage: $0 --model-key KEY [options]

Options:
  --model-key KEY       One legacy benchmark model key (required)
  --videos-dir DIR      Input video directory
  --predictions-root DIR Prediction output root
  --prompts-dir DIR     Directory containing perception/cognition/grounding/planning
  --eval-output-dir DIR Evaluation output directory
  --num-gpus N          GPU count (default: ${NUM_GPUS})
  --inference-mode MODE independent|sequential|single (default: ${INFERENCE_MODE})
  --skip-infer          Only start judge servers and evaluate existing predictions
  --skip-eval           Only run inference

Scheduling:
  Video-LLaVA/VideoChat2/Video-ChatGPT/VideoLLaMA2/VideoLLaMA3/mPLUG-Owl3
  use one model replica per GPU. MiniGPT4-Video and Tarsier2 use two GPUs
  per replica and split videos across the resulting replicas.

Environment:
  LIFEBENCH_INFER_PYTHON or LIFEBENCH_LEGACY_PYTHON overrides the legacy runtime.
EOF
}

SKIP_INFER=0
SKIP_EVAL=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-key) MODEL_KEY="${2:-}"; shift 2 ;;
        --videos-dir) VIDEOS_DIR="${2:-}"; shift 2 ;;
        --predictions-root) PREDICTIONS_ROOT="${2:-}"; shift 2 ;;
        --prompts-dir) PROMPTS_DIR="${2:-}"; shift 2 ;;
        --eval-output-dir) EVAL_OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --num-gpus) NUM_GPUS="${2:-}"; shift 2 ;;
        --inference-mode) INFERENCE_MODE="${2:-}"; shift 2 ;;
        --skip-infer) SKIP_INFER=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL_KEY" ] || [ -z "${MODEL_IDS[$MODEL_KEY]+set}" ]; then
    echo "--model-key must name one of: ${!MODEL_IDS[*]}" >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-gpus must be a positive integer" >&2
    exit 2
fi
if [ "$SKIP_INFER" -eq 1 ] && [ "$SKIP_EVAL" -eq 1 ]; then
    echo "--skip-infer and --skip-eval cannot be used together" >&2
    exit 2
fi

if [ ! -x "$INFER_PYTHON" ]; then
    echo "Inference Python not found: $INFER_PYTHON" >&2
    exit 1
fi
INFER_BIN_DIR="$(dirname "$INFER_PYTHON")"
export PATH="$INFER_BIN_DIR:$PATH"
if [ ! -x "$JUDGE_PYTHON" ]; then
    echo "Judge Python not found: $JUDGE_PYTHON" >&2
    exit 1
fi
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Videos directory not found: $VIDEOS_DIR" >&2
    exit 1
fi
if [ ! -d "$JUDGE_MODEL_PATH" ]; then
    echo "Judge model not found: $JUDGE_MODEL_PATH" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for judge server health checks" >&2
    exit 1
fi

MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${MODEL_KEY//-/_}"
SHARDED_MODEL=0
SHARD_GROUP_SIZE=2
DISABLE_PYARROW=1
case "$MODEL_KEY" in
    minigpt4-video) SHARDED_MODEL=1 ;;
    tarsier2-7b) SHARDED_MODEL=1; DISABLE_PYARROW=0 ;;
    videochat2-7b) SHARDED_MODEL=1 ;;
esac
if [ "$SHARDED_MODEL" -eq 1 ] && [ $((NUM_GPUS % SHARD_GROUP_SIZE)) -ne 0 ]; then
    echo "--num-gpus must be divisible by ${SHARD_GROUP_SIZE} for ${MODEL_KEY}" >&2
    exit 2
fi
mkdir -p "$LOG_DIR" "$PREDICTIONS_ROOT" "$EVAL_OUTPUT_DIR"
mkdir -p "$TMP_ROOT"

INFER_PIDS=()
JUDGE_PIDS=()
cleanup() {
    local pid
    for pid in "${INFER_PIDS[@]:-}" "${JUDGE_PIDS[@]:-}"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

if [ "$SKIP_INFER" -eq 0 ]; then
    echo "[1/3] ${MODEL_KEY}: inference on ${NUM_GPUS} GPUs"
    if [ "$SHARDED_MODEL" -eq 1 ]; then
        shard_count=$((NUM_GPUS / SHARD_GROUP_SIZE))
        for ((shard = 0; shard < shard_count; shard++)); do
            first_gpu=$((shard * SHARD_GROUP_SIZE))
            gpu_list=""
            for ((offset = 0; offset < SHARD_GROUP_SIZE; offset++)); do
                gpu_list="${gpu_list}$((first_gpu + offset)),"
            done
            gpu_list="${gpu_list%,}"
            log_file="${LOG_DIR}/${MODEL_KEY}_shard${shard}.log"
            CUDA_VISIBLE_DEVICES="$gpu_list" \
            LIFEBENCH_DISABLE_PYARROW="$DISABLE_PYARROW" \
            LIFEBENCH_TARSIER_ATTN_IMPLEMENTATION="${LIFEBENCH_TARSIER_ATTN_IMPLEMENTATION:-eager}" \
            LIFEBENCH_TARSIER_MAX_PIXELS="${LIFEBENCH_TARSIER_MAX_PIXELS:-50176}" \
            PYTHONFAULTHANDLER=1 \
            OMP_NUM_THREADS="$CPU_THREADS" \
            MKL_NUM_THREADS="$CPU_THREADS" \
            OPENBLAS_NUM_THREADS="$CPU_THREADS" \
            NUMEXPR_NUM_THREADS="$CPU_THREADS" \
            "$INFER_PYTHON" "${SCRIPT_DIR}/benchmark_auto_infer.py" \
                --model-keys "$MODEL_KEY" \
                --chunk-index "$shard" \
                --total-chunks "$shard_count" \
                --inference-mode "$INFERENCE_MODE" \
                --gt-hint-dir "$GT_HINT_DIR" \
                --videos-dir "$VIDEOS_DIR" \
                --output-dir "$PREDICTIONS_ROOT" \
                --prompts-dir "$PROMPTS_DIR" \
                > "$log_file" 2>&1 &
            INFER_PIDS[$shard]=$!
        done
    else
        for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
            log_file="${LOG_DIR}/${MODEL_KEY}_chunk${gpu}.log"
            CUDA_VISIBLE_DEVICES="$gpu" \
            LIFEBENCH_DISABLE_PYARROW="$DISABLE_PYARROW" \
            PYTHONFAULTHANDLER=1 \
            OMP_NUM_THREADS="$CPU_THREADS" \
            MKL_NUM_THREADS="$CPU_THREADS" \
            OPENBLAS_NUM_THREADS="$CPU_THREADS" \
            NUMEXPR_NUM_THREADS="$CPU_THREADS" \
            "$INFER_PYTHON" "${SCRIPT_DIR}/benchmark_auto_infer.py" \
                --device "$gpu" \
                --model-keys "$MODEL_KEY" \
                --chunk-index "$gpu" \
                --total-chunks "$NUM_GPUS" \
                --inference-mode "$INFERENCE_MODE" \
                --gt-hint-dir "$GT_HINT_DIR" \
                --videos-dir "$VIDEOS_DIR" \
                --output-dir "$PREDICTIONS_ROOT" \
                --prompts-dir "$PROMPTS_DIR" \
                > "$log_file" 2>&1 &
            INFER_PIDS[$gpu]=$!
        done
    fi

    infer_failed=0
    if [ "$SHARDED_MODEL" -eq 1 ]; then
        shard_count=$((NUM_GPUS / SHARD_GROUP_SIZE))
        for ((shard = 0; shard < shard_count; shard++)); do
            if wait "${INFER_PIDS[$shard]}"; then
                echo "[infer] shard ${shard} completed"
            else
                echo "[infer] shard ${shard} failed; check ${LOG_DIR}/${MODEL_KEY}_shard${shard}.log" >&2
                infer_failed=1
            fi
        done
    else
        for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
            if wait "${INFER_PIDS[$gpu]}"; then
                echo "[infer] GPU ${gpu} completed"
            else
                echo "[infer] GPU ${gpu} failed; check ${LOG_DIR}/${MODEL_KEY}_chunk${gpu}.log" >&2
                infer_failed=1
            fi
        done
    fi
    if [ "$infer_failed" -ne 0 ]; then
        echo "Inference failed; evaluation was not started." >&2
        exit 1
    fi
else
    echo "[1/3] Skipping inference"
fi

if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "[2/3] Starting Qwen3-8B judge servers"
    JUDGE_URLS=()
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        port=$((JUDGE_PORT_BASE + gpu))
        if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
            echo "[judge] Existing healthy server on port ${port}; stop it before rerunning."
            exit 1
        fi
        JUDGE_URLS+=("http://127.0.0.1:${port}/v1")
        "$JUDGE_PYTHON" "$JUDGE_SCRIPT" \
            --model-path "$JUDGE_MODEL_PATH" \
            --gpu "$gpu" \
            --port "$port" \
            > "${LOG_DIR}/judge_gpu${gpu}.log" 2>&1 &
        JUDGE_PIDS[$gpu]=$!
    done

    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        port=$((JUDGE_PORT_BASE + gpu))
        ready=0
        for ((attempt = 0; attempt < 180; attempt++)); do
            if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
                ready=1
                break
            fi
            sleep 2
        done
        if [ "$ready" -ne 1 ]; then
            echo "Judge server on GPU ${gpu}, port ${port} did not become ready" >&2
            exit 1
        fi
    done

    echo "[3/3] Evaluating ${MODEL_ID}"
    eval_command=("$JUDGE_PYTHON" "$EVAL_SCRIPT" \
        --gt-dir "$GT_DIR" \
        --predictions-root "$PREDICTIONS_ROOT" \
        --model-id "$MODEL_ID" \
        --output-dir "$EVAL_OUTPUT_DIR" \
        --judge-mode openai \
        --judge-model qwen3-8b-local \
        --judge-api-key local \
        --judge-base-urls "${JUDGE_URLS[@]}" \
        --workers "$NUM_GPUS")
    if [ "${LIFEBENCH_EVAL_RESUME:-1}" = "1" ]; then
        eval_command+=(--resume)
    fi
    if [ "${LIFEBENCH_EVAL_SKIP_TEMPORAL:-0}" = "1" ]; then
        eval_command+=(--skip-temporal)
    fi
    if [ -n "${LIFEBENCH_EVAL_RISK_STATUS_FILTER:-}" ]; then
        eval_command+=(--risk-status-filter "$LIFEBENCH_EVAL_RISK_STATUS_FILTER")
    fi
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:${PYTHONPATH}}" \
    "${eval_command[@]}"
    echo "Evaluation completed: ${EVAL_OUTPUT_DIR}"
else
    echo "[2/3] Skipping evaluation"
fi
