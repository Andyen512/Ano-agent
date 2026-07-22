#!/bin/sh

set -eu

PROJECT_ROOT="/data_4/liuyuan/lifebench"
OLD_ROOT="$PROJECT_ROOT/outputs/data_0315_top100_videofirst/benchmark_inference"
NEW_BASE="$PROJECT_ROOT/outputs/data_0315_top100_videofirst_fix2"
NEW_ROOT="$NEW_BASE/benchmark_inference"
STATUS_ROOT="$NEW_ROOT/status"
LOGS_ROOT="$NEW_ROOT/logs"
EVAL_ROOT="$NEW_BASE/evaluation/latest"
RUN_ID="data0315_top100_videofirst_fix2_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$STATUS_ROOT" "$LOGS_ROOT" "$EVAL_ROOT"

for slug in video_llava_7b videochat2_7b minigpt4_video videollama2_7b video_ccam_7b qwen2_5vl_7b mplug_owl3_7b tarsier2_7b
do
    target="$NEW_ROOT/$slug"
    if [ ! -e "$target" ]; then
        ln -s "$OLD_ROOT/$slug" "$target"
    fi
done

for status in video_llava_7b.json videochat2_7b.json minigpt4_video.json videollama2_7b.json video_ccam_7b.json qwen2_5vl_7b.json mplug_owl3_7b.json tarsier2_7b.json
do
    target="$STATUS_ROOT/$status"
    if [ ! -e "$target" ]; then
        ln -s "$OLD_ROOT/status/$status" "$target"
    fi
done

python "$PROJECT_ROOT/evaluation/benchmark_custom_dataset.py" \
    --videos-dir "$PROJECT_ROOT/data/data_0315_top100/videos_flat" \
    --prompt-file "$PROJECT_ROOT/data/infer_prompt_structured.txt" \
    --predictions-root "$NEW_ROOT" \
    --status-root "$STATUS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --models MBZUAI/Video-ChatGPT-7B DAMO-NLP-SG/VideoLLaMA3-7B \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-new-tokens 512 \
    --schedule-order video-first \
    --run-id "$RUN_ID"

python "$PROJECT_ROOT/evaluation/evaluate_outputs.py" \
    --ground-truth "$PROJECT_ROOT/data/data_0315_top100/gen_prompt_custom_top100.txt" \
    --predictions-root "$NEW_ROOT" \
    --output-dir "$EVAL_ROOT" \
    --judge-mode openai \
    --require-complete-videos \
    --expected-models \
    LanguageBind/Video-LLaVA-7B \
    OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf \
    Vision-CAIR/MiniGPT4-Video \
    DAMO-NLP-SG/VideoLLaMA2.1-7B-16F \
    OpenGVLab/InternVL3_5-8B \
    Qwen/Qwen2.5-VL-7B-Instruct \
    mPLUG/mPLUG-Owl3-7B-241101 \
    omni-research/Tarsier2-7b-0115 \
    MBZUAI/Video-ChatGPT-7B \
    DAMO-NLP-SG/VideoLLaMA3-7B

python "$PROJECT_ROOT/evaluation/render_final_results.py" \
    --summary-json "$EVAL_ROOT/summary.json" \
    --status-dir "$STATUS_ROOT" \
    --csv-out "$EVAL_ROOT/final_model_results.csv" \
    --markdown-out "$PROJECT_ROOT/FINAL_RESULTS.md"
