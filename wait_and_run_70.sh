#!/bin/bash
# 等待 GPU 空闲后自动启动 Qwen3.5-9B 70条推理 + 评测
set -e
cd /home/caiqingyuan/code/lifebench

WAIT_PID=3107193  # gait torch.distributed.launch PID

echo "[wait] 等待 gait 训练 (PID $WAIT_PID) 完成..."
while kill -0 $WAIT_PID 2>/dev/null; do
  sleep 30
done
echo "[wait] gait 训练已结束"

# 等显存释放
echo "[wait] 等待 GPU 显存释放..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 1000 ]; then
    break
  fi
  sleep 10
done
echo "[wait] GPU 已空闲"

# 杀掉 ollama 释放显存（systemd 会重启，所以循环杀）
echo "[wait] 清理 ollama..."
pkill -9 -f "ollama" 2>/dev/null || true
sleep 3

# 确认 GPU 0 有足够显存
echo "[wait] GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null

# 使用 lifebench-vlm 环境
CONDA_BASE=$(cd /home/caiqingyuan/miniconda3 && pwd)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate lifebench-vlm

# 第 1 步：推理 70 条
echo "[infer] 启动 Qwen3.5-9B 推理 70 条..."
python3 evaluation/benchmark_custom_dataset.py \
  --videos-dir /tmp/qwen35_missing_70 \
  --prompt-file data/infer_prompt_structured.txt \
  --predictions-root outputs/real_video_inference \
  --models Qwen3.5-9B \
  --gpus 0 --replicas-per-model 1
echo "[infer] 推理完成"

# 验证结果
count=$(ls outputs/real_video_inference/qwen3_5_9b/$(ls -t outputs/real_video_inference/qwen3_5_9b/ | head -1)/*.json 2>/dev/null | wc -l)
echo "[infer] 生成预测文件数: $count"

# 第 2 步：启动 Ollama 供评测使用
echo "[eval] 启动 Ollama (qwen2.5:7b)..."
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="$HOME/.ollama/models" \
  OLLAMA_KEEP_ALIVE=24h OLLAMA_VULKAN=false OLLAMA_NUM_PARALLEL=4 \
  nohup ollama serve > /tmp/ollama_eval.log 2>&1 &
echo "[eval] 等待 Ollama 启动..."
sleep 10
# 预加载模型
curl -s --max-time 120 "http://127.0.0.1:11434/api/generate" \
  -d '{"model":"qwen2.5:7b","prompt":"hi","stream":false,"keep_alive":"24h","options":{"num_predict":1}}' >/dev/null 2>&1
echo "[eval] Ollama 就绪"

# 第 3 步：评测（--resume 跳过已有 28408 条，只评测新增 70 条）
echo "[eval] 启动评测..."
python3 evaluation/evaluate_outputs.py \
  --ground-truth data/annotation/real_gt.txt \
  --predictions-root outputs/real_video_inference \
  --judge-mode openai \
  --judge-model qwen2.5:7b \
  --judge-base-url http://127.0.0.1:11434/v1 \
  --judge-api-key ollama \
  --output-dir outputs/evaluation/real_videos \
  --resume
echo "[eval] 评测完成"

# 验证
echo "[done] Qwen3.5-9B 评测条数:"
python3 -c "
import json
from collections import Counter
d=json.load(open('outputs/evaluation/real_videos/per_video_scores.json'))
c=Counter(item['model_id'] for item in d)
print('Qwen3.5-9B:', c.get('Qwen/Qwen3.5-9B',0), '(应为 2854)')
"
echo "[done] 全部完成!"
