#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/caiqingyuan/miniconda3/envs/3dpose/bin/python"
PREDICTIONS_ROOT="${LIFEBENCH_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/real_videos_single_pass}"
EVAL_OUTPUT_ROOT="${LIFEBENCH_EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation/real_videos_single_pass}"
LOG_DIR="${PROJECT_ROOT}/logs/single_pass_all_models"
FINAL_RESULTS="${PROJECT_ROOT}/outputs/evaluation/real_videos/FINAL_RESULTS_new.md"
JUDGE_PYTHON="/home/caiqingyuan/miniconda3/envs/3dpose/bin/python"
JUDGE_SCRIPT="${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/judge_server.py"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/evaluate.py"
JUDGE_MODEL_PATH="${PROJECT_ROOT}/models/Qwen__Qwen3-8B"
JUDGE_PORT_BASE="${LIFEBENCH_JUDGE_PORT_BASE:-18180}"

MODELS=(
    qwen35-9b
    internvl35-8b
    tarsier2-7b
    videollama2-7b
    qwen25vl-7b
    videollama3-7b
    video-chatgpt-7b
    videochat2-7b
)

mkdir -p "${LOG_DIR}" "${EVAL_OUTPUT_ROOT}"

run_gemini_eval() {
    local gemini_log="${LOG_DIR}/gemini-3.1-pro-eval.log"
    local gemini_output="${EVAL_OUTPUT_ROOT}/gemini_3_1_pro"
    local judge_pids=()
    local gpu port ready attempt

    if [ -f "${gemini_output}/summary.json" ] && [ -f "${gemini_output}/per_video_scores.json" ]; then
        if "${PYTHON}" - "${gemini_output}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
scores = json.loads((output_dir / "per_video_scores.json").read_text(encoding="utf-8"))
model = summary.get("model_summary", {}).get("gemini-3.1-pro", {})
sample_count = model.get("sample_count")
if not isinstance(sample_count, int) or sample_count <= 0 or sample_count != len(scores):
    raise SystemExit(1)
PY
        then
            printf '[%s] Gemini eval already complete; reusing %s\n' "$(date --iso-8601=seconds)" "${gemini_output}" | tee -a "${LOG_DIR}/orchestrator.log"
            return 0
        fi
    fi

    printf '[%s] starting Gemini eval\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_DIR}/orchestrator.log"
    : > "${gemini_log}"
    for ((gpu = 0; gpu < 8; gpu++)); do
        port=$((JUDGE_PORT_BASE + gpu))
        "${JUDGE_PYTHON}" "${JUDGE_SCRIPT}" \
            --model-path "${JUDGE_MODEL_PATH}" \
            --gpu "$gpu" \
            --port "$port" \
            >>"${gemini_log}" 2>&1 &
        judge_pids[$gpu]=$!
    done

    for ((gpu = 0; gpu < 8; gpu++)); do
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
            printf '[%s] Gemini judge on GPU %s failed to become ready\n' "$(date --iso-8601=seconds)" "$gpu" | tee -a "${gemini_log}"
            for pid in "${judge_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
            return 1
        fi
    done

    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${JUDGE_PYTHON}" "${EVAL_SCRIPT}" \
        --gt-dir "${PROJECT_ROOT}/data/public_data_release/annotations/real_videos" \
        --predictions-root "${PROJECT_ROOT}/data/public_data_release/prediction/real_videos" \
        --model-id gemini-3.1-pro \
        --output-dir "${gemini_output}" \
        --judge-mode openai \
        --judge-model qwen3-8b-local \
        --judge-api-key local \
        --judge-base-urls \
            http://127.0.0.1:18180/v1 http://127.0.0.1:18181/v1 \
            http://127.0.0.1:18182/v1 http://127.0.0.1:18183/v1 \
            http://127.0.0.1:18184/v1 http://127.0.0.1:18185/v1 \
            http://127.0.0.1:18186/v1 http://127.0.0.1:18187/v1 \
        --workers 8 \
        >>"${gemini_log}" 2>&1
    local eval_status=$?
    for pid in "${judge_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    return "$eval_status"
}

printf '[%s] starting %s models\n' "$(date --iso-8601=seconds)" "${#MODELS[@]}" | tee "${LOG_DIR}/orchestrator.log"
failed=0

for model in "${MODELS[@]}"; do
    log_file="${LOG_DIR}/${model}.log"
    printf '[%s] starting %s\n' "$(date --iso-8601=seconds)" "${model}" | tee -a "${LOG_DIR}/orchestrator.log"
    if LIFEBENCH_PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
       LIFEBENCH_EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
       "${SCRIPT_DIR}/run_model_infer_eval.sh" "${model}" >"${log_file}" 2>&1; then
        printf '[%s] %s infer/eval completed\n' "$(date --iso-8601=seconds)" "${model}" | tee -a "${LOG_DIR}/orchestrator.log"
    else
        printf '[%s] %s infer/eval failed; see %s\n' "$(date --iso-8601=seconds)" "${model}" "${log_file}" | tee -a "${LOG_DIR}/orchestrator.log"
        failed=1
    fi

    "${PYTHON}" "${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/gen_final_results_md.py" \
        --output-dir "${EVAL_OUTPUT_ROOT}" \
        --output-file "${FINAL_RESULTS}" \
        >>"${LOG_DIR}/render.log" 2>&1 || failed=1

    if [ "$model" = "qwen25vl-7b" ]; then
        run_gemini_eval || failed=1
        "${PYTHON}" "${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/gen_final_results_md.py" \
            --output-dir "${EVAL_OUTPUT_ROOT}" \
            --output-file "${FINAL_RESULTS}" \
            >>"${LOG_DIR}/render.log" 2>&1 || failed=1
    fi
done

printf '[%s] all models processed; final results: %s\n' "$(date --iso-8601=seconds)" "${FINAL_RESULTS}" | tee -a "${LOG_DIR}/orchestrator.log"
exit "${failed}"
