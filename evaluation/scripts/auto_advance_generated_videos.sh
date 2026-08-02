#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/caiqingyuan/code/lifebench"
PYTHON="/home/caiqingyuan/miniconda3/envs/3dpose/bin/python"
EVAL_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/evaluation/generated_videos"
LOG_DIR="${PROJECT_ROOT}/logs/generated_videos"
FINAL_RESULTS="${EVAL_OUTPUT_ROOT}/FINAL_RESULTS.md"

VIDEOS_DIR="${PROJECT_ROOT}/data/public_data_release/generated_videos"
PREDICTIONS_ROOT="${PROJECT_ROOT}/data/public_data_release/prediction/generated_videos"
GT_DIR="${PROJECT_ROOT}/data/public_data_release/annotations/generated_videos"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG_DIR}/auto_advance.log"
}

# model short-key -> summary path, output dir, model log, eval pid wait pattern
declare -A SUMMARY_FILE
declare -A MODEL_LOG
declare -A NEXT_MODEL

SUMMARY_FILE[videollama3_7b]="${EVAL_OUTPUT_ROOT}/videollama3_7b/summary.json"
SUMMARY_FILE[video_chatgpt_7b]="${EVAL_OUTPUT_ROOT}/video_chatgpt_7b/summary.json"
SUMMARY_FILE[videochat2_7b]="${EVAL_OUTPUT_ROOT}/videochat2_7b/summary.json"

MODEL_LOG[videollama3_7b]="${LOG_DIR}/videollama3-7b.log"
MODEL_LOG[video_chatgpt_7b]="${LOG_DIR}/video-chatgpt-7b.log"
MODEL_LOG[videochat2_7b]="${LOG_DIR}/videochat2-7b.log"

NEXT_MODEL[videollama3_7b]="video-chatgpt-7b"
NEXT_MODEL[video_chatgpt_7b]="videochat2-7b"

wait_for_summary() {
    local stage="$1"
    local summary="${SUMMARY_FILE[$stage]}"
    local model_log="${MODEL_LOG[$stage]}"
    local attempts=0
    while [ ! -f "$summary" ]; do
        attempts=$((attempts + 1))
        if [ $((attempts % 30)) -eq 0 ]; then
            local last_line=""
            if [ -f "$model_log" ]; then
                last_line=$(tail -1 "$model_log")
            fi
            log "waiting for ${stage} eval ... (${attempts} x 10s) last: ${last_line}"
        fi
        sleep 10
    done
    log "${stage} summary exists: $summary"
}

render_md() {
    if ! "$PYTHON" "${PROJECT_ROOT}/evaluation/official_evaluation/real_videos/gen_final_results_md.py" \
        --output-dir "$EVAL_OUTPUT_ROOT" \
        --output-file "$FINAL_RESULTS" \
        --title "LifeBench Generated-Videos 评测结果" \
        --evaluation-label generated_videos \
        --gt-label "data/public_data_release/annotations/generated_videos/ (13,935 条 GT)" \
        --excluded-label "gemini-3.1-pro (本次未执行)" \
        >>"${LOG_DIR}/render.log" 2>&1; then
        log "render_md FAILED"
        return 1
    fi
    log "FINAL_RESULTS.md updated"
    return 0
}

start_model() {
    local model_key="$1"
    local stage=""
    case "$model_key" in
        video-chatgpt-7b) stage="video_chatgpt_7b" ;;
        videochat2-7b) stage="videochat2_7b" ;;
        *) log "unknown model key: $model_key"; return 1 ;;
    esac
    if [ -f "${SUMMARY_FILE[$stage]}" ]; then
        log "${model_key} summary already exists, skipping launch (render only)"
        render_md
        return 0
    fi
    log "starting ${model_key} infer+eval"
    nohup env \
        LIFEBENCH_VIDEOS_DIR="$VIDEOS_DIR" \
        LIFEBENCH_PREDICTIONS_ROOT="$PREDICTIONS_ROOT" \
        LIFEBENCH_GT_DIR="$GT_DIR" \
        LIFEBENCH_EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT" \
        LIFEBENCH_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${stage}" \
        "${PROJECT_ROOT}/evaluation/scripts/run_model_infer_eval.sh" "$model_key" \
        >"${MODEL_LOG[$stage]}" 2>&1 &
    log "${model_key} launched pid $!"
}

main() {
    # Stage 1: wait for current videollama3 eval, then render + start video-chatgpt
    wait_for_summary videollama3_7b
    render_md
    start_model video-chatgpt-7b

    # Stage 2: wait for video-chatgpt, render + start videochat2
    wait_for_summary video_chatgpt_7b
    render_md
    start_model videochat2-7b

    # Stage 3: wait for final model, render once more
    wait_for_summary videochat2_7b
    render_md
    log "all models complete"
}

main
