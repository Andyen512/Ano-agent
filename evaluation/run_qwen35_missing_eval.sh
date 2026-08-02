#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JUDGE_PYTHON="${LIFEBENCH_JUDGE_PYTHON:-/home/caiqingyuan/miniconda3/envs/3dpose/bin/python}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen__Qwen3-8B"
JUDGE_SCRIPT="${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/judge_server.py"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/evaluate.py"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/evaluation/real_videos/qwen3_5_9b"

JUDGE_URLS=()
for gpu in $(seq 0 7); do
    port=$((18180 + gpu))
    JUDGE_URLS+=("http://127.0.0.1:${port}/v1")
    if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
        "$JUDGE_PYTHON" "$JUDGE_SCRIPT" \
            --model-path "$MODEL_PATH" \
            --gpu "$gpu" \
            --port "$port" \
            > "${LOG_DIR}/judge_gpu${gpu}.log" 2>&1 &
    fi
done

for gpu in $(seq 0 7); do
    port=$((18180 + gpu))
    ready=0
    for attempt in $(seq 1 180); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "judge GPU ${gpu} not ready" >&2
        exit 1
    fi
done

echo "all judge servers ready"
PYTHONPATH="$PROJECT_ROOT" "$JUDGE_PYTHON" "$EVAL_SCRIPT" \
    --gt-dir "${PROJECT_ROOT}/data/public_data_release/annotations/real_videos" \
    --predictions-root "${PROJECT_ROOT}/data/public_data_release/prediction/real_videos" \
    --model-id Qwen/Qwen3.5-9B \
    --output-dir "$OUTPUT_DIR" \
    --judge-mode openai \
    --judge-model qwen3-8b-local \
    --judge-api-key local \
    --judge-base-urls "${JUDGE_URLS[@]}" \
    --workers 8 \
    --resume
