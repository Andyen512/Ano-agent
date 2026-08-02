#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOS_DIR="${LIFEBENCH_VIDEOS_DIR:-${PROJECT_ROOT}/data/public_data_release/real_videos}"
PREDICTIONS_ROOT="${LIFEBENCH_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/real_videos}"
GT_DIR="${LIFEBENCH_GT_DIR:-${PROJECT_ROOT}/data/public_data_release/annotations/real_videos}"
PROMPTS_DIR="${LIFEBENCH_PROMPTS_DIR:-${SCRIPT_DIR}/official_evaluation/real_videos}"
EVAL_OUTPUT_DIR="${LIFEBENCH_EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/evaluation/real_videos/internvl3_5_8b}"
INFER_PYTHON="${LIFEBENCH_INFER_PYTHON:-/home/caiqingyuan/data/caiqingyuan/env/lifebench-qwen35/bin/python}"
JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-/home/caiqingyuan/miniconda3/envs/3dpose/bin/python}"
NUM_GPUS="${LIFEBENCH_NUM_GPUS:-8}"
INFER_CHUNKS=$((NUM_GPUS / 2))
JUDGE_PORT_BASE="${LIFEBENCH_JUDGE_PORT_BASE:-18180}"
CPU_THREADS="${LIFEBENCH_CPU_THREADS:-2}"
INTERNVL35_MAX_FRAMES="${LIFEBENCH_INTERNVL35_MAX_FRAMES:-32}"
LOG_DIR="$(readlink -f "${PROJECT_ROOT}/logs")"
JUDGE_SCRIPT="${SCRIPT_DIR}/official_evaluation/real_videos/judge_server.py"
EVAL_SCRIPT="${SCRIPT_DIR}/official_evaluation/real_videos/evaluate.py"
JUDGE_MODEL_PATH="${PROJECT_ROOT}/models/Qwen__Qwen3-8B"
INFER_PYTHONPATH="${PROJECT_ROOT}/.vendor/qwen35_shim${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<EOF
Usage: $0

Runs InternVL3.5-8B inference with ${INFER_CHUNKS} two-GPU workers, then evaluates
OpenGVLab/InternVL3_5-8B with the local Qwen3-8B judge cluster.

Environment overrides:
  LIFEBENCH_NUM_GPUS, LIFEBENCH_JUDGE_PORT_BASE
  LIFEBENCH_INFER_PYTHON, LIFEBENCH_JUDGE_PYTHON
  LIFEBENCH_VIDEOS_DIR, LIFEBENCH_PREDICTIONS_ROOT
  LIFEBENCH_EVAL_OUTPUT_DIR, LIFEBENCH_CPU_THREADS
  LIFEBENCH_INTERNVL35_MAX_FRAMES (default: 32)
EOF
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "Unknown argument: $1 (use --help for usage)" >&2
    exit 2
fi
if [ "$((NUM_GPUS % 2))" -ne 0 ] || [ "$INFER_CHUNKS" -lt 1 ]; then
    echo "LIFEBENCH_NUM_GPUS must be an even number >= 2 for InternVL model parallelism" >&2
    exit 1
fi

if [ ! -x "$INFER_PYTHON" ]; then
    echo "InternVL inference Python not found: $INFER_PYTHON" >&2
    exit 1
fi
if [ ! -x "$JUDGE_PYTHON" ]; then
    echo "Judge Python not found: $JUDGE_PYTHON" >&2
    exit 1
fi
if [ ! -d "${PROJECT_ROOT}/models/OpenGVLab__InternVL3_5-8B" ]; then
    echo "InternVL3.5 model not found: ${PROJECT_ROOT}/models/OpenGVLab__InternVL3_5-8B" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for judge server health checks" >&2
    exit 1
fi

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    port=$((JUDGE_PORT_BASE + gpu))
    if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
        echo "Judge server already active on GPU ${gpu}, port ${port}. Stop the existing judge cluster before inference." >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR" "$PREDICTIONS_ROOT" "$EVAL_OUTPUT_DIR"

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

echo "[1/3] Starting InternVL3.5-8B inference with ${INFER_CHUNKS} two-GPU workers"
echo "[infer] InternVL3.5 max frames: ${INTERNVL35_MAX_FRAMES}"
for ((worker = 0; worker < INFER_CHUNKS; worker++)); do
    gpu=$((worker * 2))
    peer_gpu=$((gpu + 1))
    log_file="${LOG_DIR}/internvl35_chunk${worker}.log"
    echo "[infer] GPUs ${gpu},${peer_gpu}, chunk ${worker}/${INFER_CHUNKS} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu},${peer_gpu}" \
    LIFEBENCH_DISABLE_PYARROW=1 \
    PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    NUMEXPR_NUM_THREADS="$CPU_THREADS" \
    LIFEBENCH_INTERNVL35_MAX_FRAMES="$INTERNVL35_MAX_FRAMES" \
    LIFEBENCH_INTERNVL35_MODEL_PARALLEL=1 \
    PYTHONPATH="$INFER_PYTHONPATH" \
    "$INFER_PYTHON" "${SCRIPT_DIR}/benchmark_auto_infer.py" \
        --model-keys internvl3.5-8b \
        --inference-mode single \
        --chunk-index "$worker" \
        --total-chunks "$INFER_CHUNKS" \
        --videos-dir "$VIDEOS_DIR" \
        --output-dir "$PREDICTIONS_ROOT" \
        --prompts-dir "$PROMPTS_DIR" \
        > "$log_file" 2>&1 &
    INFER_PIDS[$worker]=$!
done

INFER_FAILED=0
for ((worker = 0; worker < INFER_CHUNKS; worker++)); do
    if wait "${INFER_PIDS[$worker]}"; then
        echo "[infer] worker ${worker} completed"
    else
        echo "[infer] worker ${worker} failed; check ${LOG_DIR}/internvl35_chunk${worker}.log" >&2
        INFER_FAILED=1
    fi
done
if [ "$INFER_FAILED" -ne 0 ]; then
    echo "Inference failed; evaluation was not started." >&2
    exit 1
fi

echo "[2/3] Starting Qwen3-8B judge servers"
JUDGE_URLS=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    port=$((JUDGE_PORT_BASE + gpu))
    url="http://127.0.0.1:${port}/v1"
    JUDGE_URLS+=("$url")
    echo "[judge] GPU ${gpu}, port ${port}"
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

echo "[3/3] Evaluating InternVL3.5-8B predictions with 8 Qwen3-8B endpoints"
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:${PYTHONPATH}}" \
"$JUDGE_PYTHON" "$EVAL_SCRIPT" \
    --gt-dir "$GT_DIR" \
    --predictions-root "$PREDICTIONS_ROOT" \
    --model-id OpenGVLab/InternVL3_5-8B \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --judge-mode openai \
    --judge-model qwen3-8b-local \
    --judge-api-key local \
    --judge-base-urls "${JUDGE_URLS[@]}" \
    --workers "$NUM_GPUS" \
    --resume

echo "Inference and evaluation completed. Results: ${EVAL_OUTPUT_DIR}"
