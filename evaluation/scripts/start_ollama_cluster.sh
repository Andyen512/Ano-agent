#!/bin/bash
# scripts/start_ollama_cluster.sh - 启动 8 个 ollama 实例，分别绑 1 张 GPU
#
# 用法:
#   bash scripts/start_ollama_cluster.sh start [MODEL] [NUM_GPUS]
#   bash scripts/start_ollama_cluster.sh stop
#   bash scripts/start_ollama_cluster.sh status
#
# 示例:
#   bash scripts/start_ollama_cluster.sh start qwen2.5:7b 8
#   bash scripts/start_ollama_cluster.sh start Qwen/Qwen3-4B-Instruct 4
#
# 默认: MODEL=qwen2.5:7b, NUM_GPUS=8, BASE_PORT=11434
set -euo pipefail

MODEL="${2:-qwen2.5:7b}"
NUM_GPUS="${3:-8}"
BASE_PORT="${OLLAMA_BASE_PORT:-11434}"
LOG_DIR="${OLLAMA_LOG_DIR:-/tmp/ollama_cluster}"
PID_DIR="${OLLAMA_PID_DIR:-/tmp/ollama_cluster/pids}"
HOST_IP="${OLLAMA_HOST_IP:-127.0.0.1}"

cmd="${1:-status}"

start_one() {
    local i="$1"
    local port=$((BASE_PORT + i))
    local log_file="$LOG_DIR/ollama_gpu${i}.log"
    local pid_file="$PID_DIR/ollama_gpu${i}.pid"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "[gpu $i] already running (pid=$(cat "$pid_file"), port=$port)"
        return 0
    fi

    echo "[gpu $i] starting on port $port (log: $log_file)"
    mkdir -p "$LOG_DIR" "$PID_DIR"
    CUDA_VISIBLE_DEVICES="$i" \
    OLLAMA_HOST="$HOST_IP:$port" \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    nohup ollama serve > "$log_file" 2>&1 &
    echo $! > "$pid_file"
}

wait_ready() {
    local i="$1"
    local port=$((BASE_PORT + i))
    local url="http://$HOST_IP:$port/api/version"
    for _ in $(seq 1 30); do
        if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "[gpu $i] did not become ready within 30s; check $LOG_DIR/ollama_gpu${i}.log" >&2
    return 1
}

ensure_model() {
    local i="$1"
    local port=$((BASE_PORT + i))
    local model="$2"
    local url="http://$HOST_IP:$port"
    echo "[gpu $i] ensuring model '$model' is pulled"
    OLLAMA_HOST="$url" ollama pull "$model" >/dev/null
}

case "$cmd" in
    start)
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            start_one "$i"
        done
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            wait_ready "$i"
        done
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            ensure_model "$i" "$MODEL"
        done
        echo "ollama cluster ready: $NUM_GPUS instances on $HOST_IP:$BASE_PORT-$((BASE_PORT + NUM_GPUS - 1))"
        echo "  judge endpoints:"
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            echo "    http://$HOST_IP:$((BASE_PORT + i))/v1"
        done
        ;;

    stop)
        if [ ! -d "$PID_DIR" ]; then
            echo "no pid dir; nothing to stop"
            exit 0
        fi
        for pid_file in "$PID_DIR"/ollama_gpu*.pid; do
            [ -f "$pid_file" ] || continue
            pid="$(cat "$pid_file")"
            i="$(basename "$pid_file" .pid | sed 's/ollama_gpu//')"
            if kill -0 "$pid" 2>/dev/null; then
                echo "[gpu $i] stopping pid $pid"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        done
        echo "stopped"
        ;;

    status)
        if [ ! -d "$PID_DIR" ]; then
            echo "no pid dir; cluster not started"
            exit 0
        fi
        for pid_file in "$PID_DIR"/ollama_gpu*.pid; do
            [ -f "$pid_file" ] || continue
            pid="$(cat "$pid_file")"
            i="$(basename "$pid_file" .pid | sed 's/ollama_gpu//')"
            port=$((BASE_PORT + i))
            if kill -0 "$pid" 2>/dev/null; then
                status=$(curl -fsS -m 2 "http://$HOST_IP:$port/api/version" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null || echo "not-ready")
                echo "[gpu $i] running (pid=$pid, port=$port, version=$status)"
            else
                echo "[gpu $i] dead (stale pid=$pid)"
            fi
        done
        ;;

    *)
        echo "usage: $0 {start [MODEL] [NUM_GPUS] | stop | status}" >&2
        exit 2
        ;;
esac
