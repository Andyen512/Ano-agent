#!/bin/bash
# 启动 8 个 Ollama 实例，每卡一个，端口 11434-11441
# 用法: bash start_8ollama.sh

export OLLAMA_MODELS="$HOME/.ollama/models"

# 尝试停掉已有 ollama（systemd 的可能杀不掉，没关系，我们用不同端口）
pkill -u "$(whoami)" -f "ollama serve" 2>/dev/null || true
sleep 1

# 如果 11434 被 systemd ollama 占了，从 11435 开始
START_PORT=11434
if curl -s --max-time 2 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  START_PORT=11435
  echo "port 11434 occupied (likely systemd ollama), starting from $START_PORT"
fi

NUM_INSTANCES=$((11434 + 8 - START_PORT))
if [ "$NUM_INSTANCES" -lt 1 ]; then
  echo "all ports 11434-11441 occupied!"
  exit 1
fi

GPU_OFFSET=$((START_PORT - 11434))

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((START_PORT + i))
  gpu=$((GPU_OFFSET + i))
  CUDA_VISIBLE_DEVICES=$gpu OLLAMA_HOST="127.0.0.1:$port" OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > /tmp/ollama_$port.log 2>&1 &
  echo "started ollama on GPU $gpu, port $port (pid $!)"
  sleep 2
done

# 等待所有实例就绪
echo "waiting for instances to be ready..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((START_PORT + i))
  for retry in $(seq 1 60); do
    if curl -s --max-time 2 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
      echo "port $port ready"
      break
    fi
    sleep 1
  done
done

# 预加载模型（逐个加载，避免并发 segfault）
echo "preloading qwen2.5:7b on all instances..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((START_PORT + i))
  echo -n "loading on port $port... "
  curl -s --max-time 120 "http://127.0.0.1:$port/api/generate" \
    -d '{"model":"qwen2.5:7b","prompt":"hi","stream":false,"options":{"num_predict":1}}' >/dev/null 2>&1
  echo "done"
done

echo ""
echo "=== final status ==="
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((START_PORT + i))
  gpu=$((GPU_OFFSET + i))
  if curl -s --max-time 2 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
    echo "port $port (GPU $gpu): OK"
  else
    echo "port $port (GPU $gpu): FAILED"
  fi
done

echo ""
echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
