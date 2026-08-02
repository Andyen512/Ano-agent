#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${LIFEBENCH_JUDGE_PYTHON:-/home/caiqingyuan/miniconda3/envs/3dpose/bin/python}"
VIDEOS_DIR="${LIFEBENCH_GENERATED_VIDEOS_DIR:-${PROJECT_ROOT}/data/public_data_release/generated_videos}"
PREDICTIONS_ROOT="${LIFEBENCH_GENERATED_PREDICTIONS_ROOT:-${PROJECT_ROOT}/data/public_data_release/prediction/generated_videos}"
GT_DIR="${LIFEBENCH_GENERATED_GT_DIR:-${PROJECT_ROOT}/data/public_data_release/annotations/generated_videos}"
EVAL_OUTPUT_ROOT="${LIFEBENCH_GENERATED_EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/evaluation/generated_videos}"
LOG_DIR="${LIFEBENCH_GENERATED_LOG_DIR:-${PROJECT_ROOT}/logs/generated_videos}"
FINAL_RESULTS="${EVAL_OUTPUT_ROOT}/FINAL_RESULTS.md"

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

if [ ! -x "$PYTHON" ]; then
    echo "Judge Python not found: $PYTHON" >&2
    exit 1
fi
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Generated videos directory not found: $VIDEOS_DIR" >&2
    exit 1
fi
if [ ! -d "$GT_DIR" ]; then
    echo "Generated GT directory not found: $GT_DIR" >&2
    exit 1
fi

mkdir -p "$PREDICTIONS_ROOT" "$EVAL_OUTPUT_ROOT" "$LOG_DIR"
GLOBAL_SKIP_FILE="${PREDICTIONS_ROOT}/generated_video_infer_skip.txt"
export LIFEBENCH_GLOBAL_SKIP_FILE="$GLOBAL_SKIP_FILE"

printf '[%s] generated videos: %s\n' "$(date --iso-8601=seconds)" "$VIDEOS_DIR" | tee "$LOG_DIR/orchestrator.log"
printf '[%s] prediction root: %s\n' "$(date --iso-8601=seconds)" "$PREDICTIONS_ROOT" | tee -a "$LOG_DIR/orchestrator.log"
printf '[%s] GT directory: %s\n' "$(date --iso-8601=seconds)" "$GT_DIR" | tee -a "$LOG_DIR/orchestrator.log"

SCHEMA_REPORT="${EVAL_OUTPUT_ROOT}/gt_schema_comparison.json"
if "$PYTHON" "${SCRIPT_DIR}/compare_gt_schema.py" \
    --real-gt-dir "${PROJECT_ROOT}/data/public_data_release/annotations/real_videos" \
    --generated-gt-dir "$GT_DIR" \
    --output "$SCHEMA_REPORT" \
    >"${LOG_DIR}/gt_schema_comparison.log" 2>&1; then
    printf '[%s] generated GT fields match real-video GT schema; report: %s\n' \
        "$(date --iso-8601=seconds)" "$SCHEMA_REPORT" | tee -a "$LOG_DIR/orchestrator.log"
else
    printf '[%s] generated GT schema differs from real-video GT schema; see %s\n' \
        "$(date --iso-8601=seconds)" "${LOG_DIR}/gt_schema_comparison.log" | tee -a "$LOG_DIR/orchestrator.log"
    exit 1
fi

failed=0
for model in "${MODELS[@]}"; do
    log_file="${LOG_DIR}/${model}.log"
    case "$model" in
        qwen35-9b) rebalance_model_key="qwen3.5-9b" ;;
        internvl35-8b) rebalance_model_key="internvl3.5-8b" ;;
        qwen25vl-7b) rebalance_model_key="qwen2.5vl-7b" ;;
        *) rebalance_model_key="$model" ;;
    esac
    printf '[%s] starting %s\n' "$(date --iso-8601=seconds)" "$model" | tee -a "$LOG_DIR/orchestrator.log"
    if LIFEBENCH_VIDEOS_DIR="$VIDEOS_DIR" \
       LIFEBENCH_PREDICTIONS_ROOT="$PREDICTIONS_ROOT" \
       LIFEBENCH_GT_DIR="$GT_DIR" \
       LIFEBENCH_EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT" \
       LIFEBENCH_REBALANCE_PENDING=1 \
       LIFEBENCH_REBALANCE_MODEL_KEY="$rebalance_model_key" \
       LIFEBENCH_INFERENCE_MODE=single \
       bash "${SCRIPT_DIR}/run_model_infer_eval.sh" "$model" >"$log_file" 2>&1; then
        printf '[%s] %s completed\n' "$(date --iso-8601=seconds)" "$model" | tee -a "$LOG_DIR/orchestrator.log"
    else
        printf '[%s] %s failed; see %s\n' "$(date --iso-8601=seconds)" "$model" "$log_file" | tee -a "$LOG_DIR/orchestrator.log"
         failed=1
    fi

    if [ "$model" = "qwen35-9b" ] && [ "$failed" -eq 0 ]; then
        if ! "$PYTHON" "${SCRIPT_DIR}/check_generated_eval_complete.py" \
            --gt-dir "$GT_DIR" \
            --predictions-root "$PREDICTIONS_ROOT" \
            --summary "${EVAL_OUTPUT_ROOT}/qwen3_5_9b/summary.json" \
            --model-id "Qwen/Qwen3.5-9B" | tee -a "$LOG_DIR/orchestrator.log"; then
            printf '[%s] Qwen3.5 evaluation incomplete; stopping before next model\n' \
                "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/orchestrator.log"
            failed=1
            break
        fi
    fi

    if [ "$failed" -eq 0 ]; then
        if ! "$PYTHON" "${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/gen_final_results_md.py" \
            --output-dir "$EVAL_OUTPUT_ROOT" \
            --output-file "$FINAL_RESULTS" \
            --title "LifeBench Generated-Videos 评测结果" \
            --evaluation-label generated_videos \
            --gt-label "data/public_data_release/annotations/generated_videos/ (13,935 条 GT)" \
            --excluded-label "gemini-3.1-pro (本次未执行)" \
            >>"$LOG_DIR/render.log" 2>&1; then
            printf '[%s] %s final-results rendering failed; stopping before next model\n' \
                "$(date --iso-8601=seconds)" "$model" | tee -a "$LOG_DIR/orchestrator.log"
            failed=1
            break
        fi
        printf '[%s] %s final results updated: %s\n' \
            "$(date --iso-8601=seconds)" "$model" "$FINAL_RESULTS" | tee -a "$LOG_DIR/orchestrator.log"
    fi
done

if "$PYTHON" "${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/gen_final_results_md.py" \
    --output-dir "$EVAL_OUTPUT_ROOT" \
    --output-file "$FINAL_RESULTS" \
    --title "LifeBench Generated-Videos 评测结果" \
    --evaluation-label generated_videos \
    --gt-label "data/public_data_release/annotations/generated_videos/ (13,935 条 GT)" \
    --excluded-label "gemini-3.1-pro (本次未执行)" \
    >>"$LOG_DIR/render.log" 2>&1; then
    printf '[%s] final results: %s\n' "$(date --iso-8601=seconds)" "$FINAL_RESULTS" | tee -a "$LOG_DIR/orchestrator.log"
else
    printf '[%s] final results rendering failed; see %s\n' "$(date --iso-8601=seconds)" "$LOG_DIR/render.log" | tee -a "$LOG_DIR/orchestrator.log"
    failed=1
fi

exit "$failed"
