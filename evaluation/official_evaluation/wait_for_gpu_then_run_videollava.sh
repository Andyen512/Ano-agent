#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/caiqingyuan/code/lifebench"
PYTHON="/home/caiqingyuan/miniconda3/envs/3dpose/bin/python"
MODEL_PATH="${PROJECT_ROOT}/models/LanguageBind__Video-LLaVA-7B"
LOG_DIR="${PROJECT_ROOT}/outputs/evaluation/video-llava-7b-wait"
INFER_LOG="${LOG_DIR}/infer.log"
EVAL_LOG="${LOG_DIR}/eval.log"
WAIT_LOG="${LOG_DIR}/wait.log"

mkdir -p "${LOG_DIR}"
printf '[%s] waiting for GPU 0 to become available\n' "$(date --iso-8601=seconds)" >> "${WAIT_LOG}"

while true; do
    gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | python -c 'import sys; print(next((line.strip() for line in sys.stdin if line.strip()), "999999"))')"
    if [ "${gpu_used}" -le 1000 ] 2>/dev/null; then
        printf '[%s] GPU 0 usage is %s MiB; starting inference\n' "$(date --iso-8601=seconds)" "${gpu_used}" >> "${WAIT_LOG}"
        break
    fi
    printf '[%s] GPU 0 usage is %s MiB; retrying in 60 seconds\n' "$(date --iso-8601=seconds)" "${gpu_used}" >> "${WAIT_LOG}"
    sleep 60
done

cd "${PROJECT_ROOT}" || exit 1
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" evaluation/official_evaluation/videollava_infer.py \
    --model-path "${MODEL_PATH}" \
    --max-new-tokens 256 \
    >"${INFER_LOG}" 2>&1
infer_status=$?


"${PYTHON}" evaluation/official_evaluation/videollava_eval.py \
    --judge-mode heuristic \
    --workers 8 \
    >"${EVAL_LOG}" 2>&1
eval_status=$?

printf '[%s] evaluation exited with code %s\n' "$(date --iso-8601=seconds)" "${eval_status}" >> "${WAIT_LOG}"
if [ "${infer_status}" -ne 0 ]; then
    exit "${infer_status}"
fi
exit "${eval_status}"
