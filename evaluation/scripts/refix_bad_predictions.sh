#!/bin/bash
# 修复 video_llava_7b 和 mplug_owl3_7b 中缺失或输出为空/截断的预测文件
# Usage: bash evaluation/scripts/refix_bad_predictions.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/evaluation"
PRED_ROOT="${PROJECT_ROOT}/data/public_data_release/prediction/real_videos"
PROMPT_FILE="${PROJECT_ROOT}/data/infer_prompt_normal_desc.txt"
TIMESTAMP="20260712T040531Z"
MAX_NEW_TOKENS=1024

MODELS=("video_llava_7b" "mplug_owl3_7b")
# 参考模型：从它获取 video_path（输出完整且正确）
REF_MODEL="videollama3_7b"
REF_TIMESTAMP="20260713T103610Z"

declare -A BACKEND MODEL_ID MODEL_PATH GPU
BACKEND["video_llava_7b"]="videollava"
MODEL_ID["video_llava_7b"]="LanguageBind/Video-LLaVA-7B"
MODEL_PATH["video_llava_7b"]="${PROJECT_ROOT}/models/LanguageBind__Video-LLaVA-7B"
GPU["video_llava_7b"]="${GPU_VIDEO_LLAVA:-5}"

BACKEND["mplug_owl3_7b"]="mplugowl3"
MODEL_ID["mplug_owl3_7b"]="mPLUG/mPLUG-Owl3-7B-241101"
MODEL_PATH["mplug_owl3_7b"]="${PROJECT_ROOT}/models/mPLUG__mPLUG-Owl3-7B-241101"
GPU["mplug_owl3_7b"]="${GPU_MPLUG_OWL3:-6}"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Step 1: 从参考模型建立 stem -> video_path 索引 ==="
python3 << PYEOF > "${TMPDIR}/ref_index.json"
import sys, json, os
from pathlib import Path

ref_dir = Path('${PRED_ROOT}/${REF_MODEL}/${REF_TIMESTAMP}')
index = {}

for f in sorted(ref_dir.glob('*.json')):
    try:
        payload = json.loads(f.read_text(encoding='utf-8'))
        video_path = payload.get('video_path', '')
        if video_path:
            index[f.stem] = video_path
    except Exception:
        pass

json.dump(index, sys.stdout, ensure_ascii=False)
PYEOF

INDEX_COUNT=$(python3 -c "import json; print(len(json.load(open('${TMPDIR}/ref_index.json'))))")
echo "  Indexed ${INDEX_COUNT} video_paths from ${REF_MODEL}"

echo ""
echo "=== Step 2: 扫描缺失/损坏的预测文件，生成 task manifest ==="

for model in "${MODELS[@]}"; do
  MODEL_PRED_DIR="${PRED_ROOT}/${model}/${TIMESTAMP}"
  TASK_MANIFEST="${TMPDIR}/${model}_tasks.json"

  mkdir -p "$MODEL_PRED_DIR"

  python3 << PYEOF > "$TASK_MANIFEST"
import sys, json, os
sys.path.insert(0, '${PROJECT_ROOT}')
from evaluation.official_evaluation.real_videos.common_real import parse_prediction_response
from pathlib import Path

# 加载参考索引
with open('${TMPDIR}/ref_index.json') as f:
    ref_index = json.load(f)

pred_dir = Path('${MODEL_PRED_DIR}')
existing = set()
bad = set()

# 扫描现有文件
for f in sorted(pred_dir.glob('*.json')):
    existing.add(f.stem)
    try:
        payload = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        bad.add(f.stem)
        continue
    if parse_prediction_response(payload) is None:
        bad.add(f.stem)

# 修正：确保用参考模型检查的 stem 集
# 只处理在参考模型中存在的 stem（说明是有效视频）
tasks = []
for stem, video_path in ref_index.items():
    if stem not in existing or stem in bad:
        tasks.append({
            "video_path": video_path,
            "output_json": os.path.join('${MODEL_PRED_DIR}', f"{stem}.json"),
            "stdout_log": os.path.join('${MODEL_PRED_DIR}', f"{stem}.stdout.log"),
            "stderr_log": os.path.join('${MODEL_PRED_DIR}', f"{stem}.stderr.log"),
        })

json.dump(tasks, sys.stdout, ensure_ascii=False)
print(f"\n{len(tasks)} tasks (existing={len(existing)}, bad={len(bad)}, ref={len(ref_index)})", file=sys.stderr)
PYEOF

  TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('$TASK_MANIFEST'))))")
  echo "  [${model}] ${TASK_COUNT} tasks to re-infer"
done

# ========================
# Step 3: 双 GPU 并行推理
# ========================
echo ""
echo "=== Step 3: 并行推理 ==="
PIDS=()

for model in "${MODELS[@]}"; do
  TASK_MANIFEST="${TMPDIR}/${model}_tasks.json"
  TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('$TASK_MANIFEST'))))")
  if [ "$TASK_COUNT" -eq 0 ]; then
    echo "  [${model}] No tasks, skip."
    continue
  fi

  local_gpu="${GPU[$model]}"
  echo "  [${model}] GPU ${local_gpu}: ${TASK_COUNT} tasks"

  (
    source "${PROJECT_ROOT}/activate_lifebench_vlm.sh"
    CUDA_VISIBLE_DEVICES="${local_gpu}" python "${EVAL_DIR}/lifebench_batch_infer.py" \
      --backend "${BACKEND[$model]}" \
      --model-id "${MODEL_ID[$model]}" \
      --model-path "${MODEL_PATH[$model]}" \
      --prompt-file "$PROMPT_FILE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --skip-existing \
      --task-manifest "$TASK_MANIFEST" \
      --summary-json "${TMPDIR}/${model}_summary.json" \
      2>&1 | tee "${TMPDIR}/${model}_infer.log"
    echo "[$(date)] ${model}: DONE"
  ) &
  PIDS+=($!)
  echo "  [${model}] PID: ${PIDS[-1]}"
done

echo ""
echo "Waiting for all GPUs ..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo ""
echo "=== All done ==="
