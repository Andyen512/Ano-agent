#!/bin/bash
# ============================================================================
# run_benchmark_auto_infer_parallel.sh - 8 卡并行推理
#
# 用法:
#   ./run_benchmark_auto_infer_parallel.sh \
#       --videos-dir data/public_data_release/real_videos \
#       --output-dir data/public_data_release/prediction
#
# 将 11 个模型均匀分配到 8 张 GPU 上并行推理。
# ============================================================================

set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Avoid the libarrow/jemalloc background-thread segfault seen with the
# Qwen3.5 processing stack in this environment.
export ARROW_DEFAULT_MEMORY_POOL=system
# libarrow 24.0.0 still starts jemalloc's background thread in this runtime;
# disable it because that thread is the source of the observed SIGSEGV.
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES="${LIFEBENCH_TORCH_SHOW_CPP_STACKTRACES:-1}"
export CUDA_LAUNCH_BLOCKING="${LIFEBENCH_CUDA_LAUNCH_BLOCKING:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$PARENT_DIR/activate_lifebench_vlm.sh" >/dev/null

# 所有模型 key（保证顺序与 MODELS 列表一致）
ALL_KEYS=(
    "video-llava-7b"
    # "videochat2-7b"    # OOM with 32 frames
    "video-chatgpt-7b"
    "minigpt4-video"
    "videollama2-7b"
    "qwen2.5vl-7b"
    "qwen3.5-9b"
    "videollama3-7b"
    "mplug-owl3-7b"
    # "internvl3.5-8b"   # OOM on 24GB GPU
    # "tarsier2-7b"      # disabled
)
NUM_GPUS=8

# A worker can disappear because of a transient CUDA/runtime failure.  The
# Python pipeline persists progress in --output-dir, so restarting a worker
# resumes unfinished dimensions instead of repeating completed work.
MAX_RETRIES="${LIFEBENCH_INFER_MAX_RETRIES:-3}"
RETRY_DELAY="${LIFEBENCH_INFER_RETRY_DELAY:-10}"

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ && "$RETRY_DELAY" =~ ^[0-9]+$ ]]; then
    echo "LIFEBENCH_INFER_MAX_RETRIES and LIFEBENCH_INFER_RETRY_DELAY must be non-negative integers" >&2
    exit 1
fi

# 解析公共参数（--videos-dir, --output-dir, --prompts-dir, --model-keys,
# --inference-mode, --chunk-index, --total-chunks, --gpu-index）
COMMON_ARGS=()
USER_KEYS=()
TARGET_GPU=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --videos-dir|--output-dir|--prompts-dir|--inference-mode|--chunk-index|--total-chunks)
            COMMON_ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpu-index)
            TARGET_GPU="$2"
            shift 2
            ;;
        --model-keys)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                USER_KEYS+=("$1")
                shift
            done
            ;;
        *)
             echo "Usage: $0 [--videos-dir DIR] [--output-dir DIR] [--prompts-dir DIR] [--model-keys KEY...] [--inference-mode independent|sequential|single] [--chunk-index N --total-chunks N] [--gpu-index N]"
            exit 1
            ;;
    esac
done

CHUNK_INDEX=""
TOTAL_CHUNKS=""
for ((arg_i = 0; arg_i < ${#COMMON_ARGS[@]}; arg_i += 2)); do
    case "${COMMON_ARGS[arg_i]}" in
        --chunk-index) CHUNK_INDEX="${COMMON_ARGS[arg_i + 1]}" ;;
        --total-chunks) TOTAL_CHUNKS="${COMMON_ARGS[arg_i + 1]}" ;;
    esac
done
if [ -n "$CHUNK_INDEX" ] || [ -n "$TOTAL_CHUNKS" ]; then
    if ! [[ "$CHUNK_INDEX" =~ ^[0-9]+$ && "$TOTAL_CHUNKS" =~ ^[1-9][0-9]*$ ]] || \
        [ "$CHUNK_INDEX" -ge "$TOTAL_CHUNKS" ]; then
        echo "--chunk-index must be in [0, --total-chunks), and --total-chunks must be positive" >&2
        exit 1
    fi
fi
if [ -n "$TARGET_GPU" ] && ! [[ "$TARGET_GPU" =~ ^[0-9]+$ && "$TARGET_GPU" -lt "$NUM_GPUS" ]]; then
    echo "--gpu-index must be an integer in [0, ${NUM_GPUS})" >&2
    exit 1
fi

echo "============================================"
echo " LifeBench 8-GPU Parallel Inference"
echo "============================================"
LOG_DIR="$(readlink -f "${SCRIPT_DIR}/../logs")"
mkdir -p "${LOG_DIR}"

if [ ${#USER_KEYS[@]} -gt 0 ]; then
    EFFECTIVE_KEYS=("${USER_KEYS[@]}")
else
    EFFECTIVE_KEYS=("${ALL_KEYS[@]}")
fi

echo "Models:     ${#EFFECTIVE_KEYS[@]} total"
echo "GPUs:       ${NUM_GPUS}"
echo "Args:       ${COMMON_ARGS[*]:-(defaults)}"
echo "Log dir:    ${LOG_DIR}"
echo ""

# 分配模型到 GPU：先每 GPU 分 1 个，剩余的从 GPU 0 开始依次加
ASSIGN=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    ASSIGN[gpu]=""
done

gpu=0
[ -n "$TARGET_GPU" ] && gpu="$TARGET_GPU"
for key in "${EFFECTIVE_KEYS[@]}"; do
    ASSIGN[gpu]="${ASSIGN[gpu]} $key"
    gpu=$(( (gpu + 1) % NUM_GPUS ))
done

# 启动进程。每个 GPU 都有独立的重试计数；某张卡失败不会阻断其他卡。
PIDS=()
RETRIES=()
launch_worker() {
    local gpu="$1"
    local keys="$2"
    local attempt="${RETRIES[$gpu]:-0}"
    local log_file="${LOG_DIR}/infer_gpu${gpu}.log"
    local worker_pythonpath="${PYTHONPATH:-}"
    local disable_pyarrow="0"
    local worker_python="${LIFEBENCH_PYTHON:-python}"

    # benchmark_auto_infer imports transformers during module startup. Put the
    # Qwen3.5 shim on PYTHONPATH before Python starts, otherwise the process
    # can mix the global Transformers install with the isolated Qwen runtime.
    if [[ " $keys " == *" qwen3.5-9b "* ]]; then
        worker_pythonpath="${PARENT_DIR}/.vendor/qwen35_shim${worker_pythonpath:+:${worker_pythonpath}}"
        disable_pyarrow="1"
        worker_python="${LIFEBENCH_QWEN_PYTHON:-$worker_python}"
    fi

    {
        echo ""
        echo "========== GPU ${gpu}, worker attempt $((attempt + 1)) =========="
    } >> "$log_file"
    LIFEBENCH_DISABLE_PYARROW="$disable_pyarrow" \
        PYTHONPATH="$worker_pythonpath" "$worker_python" "${SCRIPT_DIR}/benchmark_auto_infer.py" \
        --device "$gpu" \
        --model-keys $keys \
        "${COMMON_ARGS[@]}" \
        >> "$log_file" 2>&1 &
    PIDS[$gpu]=$!
}

cleanup_workers() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup_workers INT TERM

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    if [ -n "$TARGET_GPU" ] && [ "$gpu" -ne "$TARGET_GPU" ]; then
        continue
    fi
    keys="${ASSIGN[gpu]}"
    if [ -z "$keys" ]; then
        continue
    fi
    echo "[GPU ${gpu}] models:${keys}"
    RETRIES[$gpu]=0
    launch_worker "$gpu" "$keys"
done

echo ""
echo "All GPU processes launched. Waiting for completion..."
echo "Logs: ${LOG_DIR}/infer_gpu*.log"
echo ""

# 等待所有进程完成；失败的 worker 在达到上限前自动重启。
FAILED=0
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    pid="${PIDS[$gpu]:-}"
    if [ -z "$pid" ]; then
        continue
    fi

    while true; do
        if wait "$pid"; then
            rc=0
        else
            rc=$?
        fi

        if [ "$rc" -ge 128 ]; then
            signal=$((rc - 128))
            exit_detail="signal ${signal}"
        else
            exit_detail="exit code ${rc}"
        fi
        echo "[GPU ${gpu}] worker attempt $((RETRIES[$gpu] + 1)) ended with ${exit_detail}" \
            >> "${LOG_DIR}/infer_gpu${gpu}.log"

        if [ "$rc" -eq 0 ]; then
            echo "[GPU ${gpu}] completed"
            break
        fi

        if [ "${RETRIES[$gpu]:-0}" -ge "$MAX_RETRIES" ]; then
            echo "[GPU ${gpu}] FAILED after $((MAX_RETRIES + 1)) attempts (last exit code ${rc})"
            FAILED=1
            break
        fi

        RETRIES[$gpu]=$((RETRIES[$gpu] + 1))
        echo "[GPU ${gpu}] exited with code ${rc}; retry ${RETRIES[$gpu]}/${MAX_RETRIES} in ${RETRY_DELAY}s"
        sleep "$RETRY_DELAY"
        launch_worker "$gpu" "${ASSIGN[gpu]}"
        pid="${PIDS[$gpu]}"
    done
done

echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo " All GPU processes completed successfully!"
else
    echo " Some GPU processes failed. Check logs for details."
fi
echo "============================================"
exit $FAILED
