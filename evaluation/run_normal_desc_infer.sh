#!/bin/bash
# 8个模型在 4 张 GPU 上并行推理 (每卡顺序执行，卡间并行)
# 每个模型只推理该模型之前判定为 normal 的视频
# Usage: bash run_normal_desc_infer.sh
set -euo pipefail

GPUS=(2 5 6 7)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/evaluation"
MANIFEST_DIR="${PROJECT_ROOT}/data/public_data_release/prediction/real_videos_normal_desc"
PROMPT_FILE="${PROJECT_ROOT}/data/infer_prompt_normal_desc.txt"

# ============================================================
# Model registry
# ============================================================
declare -A BACKEND MODEL_ID MODEL_PATH VIDEO_COUNT

BACKEND["videollama3_7b"]="videollama3"
MODEL_ID["videollama3_7b"]="DAMO-NLP-SG/VideoLLaMA3-7B"
MODEL_PATH["videollama3_7b"]="${PROJECT_ROOT}/models/DAMO-NLP-SG__VideoLLaMA3-7B"
VIDEO_COUNT["videollama3_7b"]=2543

BACKEND["mplug_owl3_7b"]="mplugowl3"
MODEL_ID["mplug_owl3_7b"]="mPLUG/mPLUG-Owl3-7B-241101"
MODEL_PATH["mplug_owl3_7b"]="${PROJECT_ROOT}/models/mPLUG__mPLUG-Owl3-7B-241101"
VIDEO_COUNT["mplug_owl3_7b"]=2372

BACKEND["video_llava_7b"]="videollava"
MODEL_ID["video_llava_7b"]="LanguageBind/Video-LLaVA-7B"
MODEL_PATH["video_llava_7b"]="${PROJECT_ROOT}/models/LanguageBind__Video-LLaVA-7B"
VIDEO_COUNT["video_llava_7b"]=2257

BACKEND["qwen2.5vl_7b"]="qwen25vl"
MODEL_ID["qwen2.5vl_7b"]="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH["qwen2.5vl_7b"]="${PROJECT_ROOT}/models/Qwen__Qwen2.5-VL-7B-Instruct"
VIDEO_COUNT["qwen2.5vl_7b"]=1713

BACKEND["video_chatgpt_7b"]="videochatgpt"
MODEL_ID["video_chatgpt_7b"]="MBZUAI/Video-ChatGPT-7B"
MODEL_PATH["video_chatgpt_7b"]="${PROJECT_ROOT}/models/MBZUAI__Video-ChatGPT-7B"
VIDEO_COUNT["video_chatgpt_7b"]=1037

BACKEND["qwen3.5_9b"]="qwen35vl"
MODEL_ID["qwen3.5_9b"]="Qwen/Qwen3.5-9B"
MODEL_PATH["qwen3.5_9b"]="${PROJECT_ROOT}/models/Qwen/Qwen3.5-9B"
VIDEO_COUNT["qwen3.5_9b"]=975

BACKEND["videochat2_7b"]="videochat2"
MODEL_ID["videochat2_7b"]="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf"
MODEL_PATH["videochat2_7b"]="${PROJECT_ROOT}/models/OpenGVLab__VideoChat2_HD_stage4_Mistral_7B_hf"
VIDEO_COUNT["videochat2_7b"]=866

BACKEND["videollama2_7b"]="videollama2"
MODEL_ID["videollama2_7b"]="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
MODEL_PATH["videollama2_7b"]="${PROJECT_ROOT}/models/DAMO-NLP-SG__VideoLLaMA2.1-7B-16F"
VIDEO_COUNT["videollama2_7b"]=374

# ============================================================
# Balanced assignment (by video count)
# GPU 2: 2917 | GPU 5: 3238 | GPU 6: 3232 | GPU 7: 2750
# ============================================================
GPU_MODELS_2=("videollama3_7b" "videollama2_7b")
GPU_MODELS_5=("mplug_owl3_7b")
GPU_MODELS_6=("video_llava_7b" "qwen3.5_9b")
GPU_MODELS_7=("qwen2.5vl_7b" "video_chatgpt_7b")

declare -A GPU_MODELS
GPU_MODELS[2]="GPU_MODELS_2[@]"
GPU_MODELS[5]="GPU_MODELS_5[@]"
GPU_MODELS[6]="GPU_MODELS_6[@]"
GPU_MODELS[7]="GPU_MODELS_7[@]"

# ============================================================
# Run models on one GPU (sequential, each model failure does not block next)
# ============================================================
run_models_on_gpu() {
  local gpu=$1; shift
  local models=("$@")
  local logfile="${MANIFEST_DIR}/gpu${gpu}_run.log"

  echo "[$(date)] GPU ${gpu}: starting ${#models[@]} models" | tee -a "$logfile"
  for mk in "${models[@]}"; do
    local backend="${BACKEND[$mk]}"
    local model_id="${MODEL_ID[$mk]}"
    local model_path="${MODEL_PATH[$mk]}"
    local n_videos="${VIDEO_COUNT[$mk]}"
    local manifest="${MANIFEST_DIR}/${mk}_manifest.json"
    local summary="${MANIFEST_DIR}/${mk}_summary.json"

    if [ ! -f "$manifest" ]; then
      echo "[$(date)] GPU ${gpu}: [${mk}] manifest not found, SKIP" | tee -a "$logfile"
      continue
    fi

    echo "[$(date)] GPU ${gpu}: [${mk}] loading (${n_videos} videos)..." | tee -a "$logfile"

    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python "${EVAL_DIR}/lifebench_batch_infer.py" \
      --backend "$backend" \
      --model-id "$model_id" \
      --model-path "$model_path" \
      --prompt-file "$PROMPT_FILE" \
      --max-new-tokens 512 \
      --skip-existing \
      --task-manifest "$manifest" \
      --summary-json "$summary" \
      2>&1 | tee -a "$logfile"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
      echo "[$(date)] GPU ${gpu}: [${mk}] DONE" | tee -a "$logfile"
    else
      echo "[$(date)] GPU ${gpu}: [${mk}] FAILED (exit ${rc})" | tee -a "$logfile"
    fi
  done
  echo "[$(date)] GPU ${gpu}: all done" | tee -a "$logfile"
}

# ============================================================
# Print summary
# ============================================================
echo "============================================================"
echo " Normal Video Description Re-Inference (8 models × 4 GPUs)"
echo "============================================================"
echo ""

for gpu in "${GPUS[@]}"; do
  models_ref="${GPU_MODELS[$gpu]}"
  models_arr=("${!models_ref}")
  total=0
  for m in "${models_arr[@]}"; do ((total += ${VIDEO_COUNT[$m]:-0})); done
  printf "GPU %s (%d videos):\n" "$gpu" "$total"
  for m in "${models_arr[@]}"; do
    printf "  %-20s %d\n" "$m" "${VIDEO_COUNT[$m]:-0}"
  done
  echo ""
done

# Activate environment
source "${PROJECT_ROOT}/activate_lifebench_vlm.sh"

# Launch all 4 GPU workers in parallel
PIDS=()
for gpu in "${GPUS[@]}"; do
  models_ref="${GPU_MODELS[$gpu]}"
  models_arr=("${!models_ref}")
  run_models_on_gpu "$gpu" "${models_arr[@]}" &
  PIDS+=($!)
  echo "GPU ${gpu} PID: ${PIDS[-1]}"
done

echo ""
echo "Waiting for all GPUs to finish..."
echo "Progress: tail -f ${MANIFEST_DIR}/gpu*.run.log"
echo ""

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE."
echo "============================================================"
