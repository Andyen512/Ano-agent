#!/bin/bash
# scripts/run_evaluation.sh - 评测入口，自动检测本地 ollama cluster 端口并拼好命令
#
# 用法:
#   bash scripts/run_evaluation.sh [extra args to run_benchmark_evaluation.sh]
#
# 环境变量:
#   EVAL_JUDGE_MODEL          judge 模型名 (默认 qwen2.5:7b)
#   EVAL_JUDGE_API_KEY        judge API key (默认 ollama)
#   EVAL_JUDGE_TIMEOUT_SECONDS  judge 请求超时秒 (默认 180)
#   EVAL_NUM_GPUS             并发 GPU 数 (默认 8; 设为 1 表示单 ollama)
#   EVAL_BASE_PORT            ollama cluster 起始端口 (默认 11434)
#   EVAL_HOST                 ollama host (默认 127.0.0.1)
#   EVAL_WORKERS              评测 worker 线程数 (默认 = EVAL_NUM_GPUS)
#   EVAL_GT_PATH              ground truth 路径
#   EVAL_PREDICTIONS_ROOT     predictions 根目录
#   EVAL_OUTPUT_DIR           评测输出目录
#   EVAL_REQUIRE_COMPLETE     是否 --require-complete-videos (默认 1)
#   EVAL_EXPECTED_MODELS      显式列出的模型 id，空格分隔 (默认 10 个 benchmark 模型)
#
# 示例:
#   bash scripts/run_evaluation.sh
#   EVAL_NUM_GPUS=4 bash scripts/run_evaluation.sh
#   EVAL_NUM_GPUS=1 EVAL_OUTPUT_DIR=/tmp/eval1 bash scripts/run_evaluation.sh --model-id Qwen/Qwen3.5-9B
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:-qwen2.5:7b}"
EVAL_JUDGE_API_KEY="${EVAL_JUDGE_API_KEY:-ollama}"
EVAL_JUDGE_TIMEOUT_SECONDS="${EVAL_JUDGE_TIMEOUT_SECONDS:-180}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-8}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-11434}"
EVAL_HOST="${EVAL_HOST:-127.0.0.1}"
EVAL_WORKERS="${EVAL_WORKERS:-$EVAL_NUM_GPUS}"
EVAL_GT_PATH="${EVAL_GT_PATH:-$PROJECT_ROOT/data/annotation/gen_prompt_generated_all.txt}"
EVAL_PREDICTIONS_ROOT="${EVAL_PREDICTIONS_ROOT:-$PROJECT_ROOT/outputs/predictions}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PROJECT_ROOT/outputs/evaluation/generated_videos}"
EVAL_REQUIRE_COMPLETE="${EVAL_REQUIRE_COMPLETE:-1}"
EVAL_EXPECTED_MODELS_DEFAULT=(
    "LanguageBind/Video-LLaVA-7B"
    "OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf"
    "MBZUAI/Video-ChatGPT-7B"
    "Vision-CAIR/MiniGPT4-Video"
    "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen3.5-9B"
    "DAMO-NLP-SG/VideoLLaMA3-7B"
    "mPLUG/mPLUG-Owl3-7B-241101"
    "omni-research/Tarsier2-7b-0115"
)
if [ -n "${EVAL_EXPECTED_MODELS:-}" ]; then
    # shellcheck disable=SC2206
    EXPECTED_MODELS=($EVAL_EXPECTED_MODELS)
else
    EXPECTED_MODELS=("${EVAL_EXPECTED_MODELS_DEFAULT[@]}")
fi

# Build judge endpoints
ENDPOINTS=()
for i in $(seq 0 $((EVAL_NUM_GPUS - 1))); do
    port=$((EVAL_BASE_PORT + i))
    ENDPOINTS+=("http://$EVAL_HOST:$port/v1")
done

echo "== evaluation config =="
echo "  GT path:           $EVAL_GT_PATH"
echo "  predictions root:  $EVAL_PREDICTIONS_ROOT"
echo "  output dir:        $EVAL_OUTPUT_DIR"
echo "  judge model:       $EVAL_JUDGE_MODEL"
echo "  judge timeout:     ${EVAL_JUDGE_TIMEOUT_SECONDS}s"
echo "  workers:           $EVAL_WORKERS"
echo "  gpu endpoints:     ${#ENDPOINTS[@]}"
for ep in "${ENDPOINTS[@]}"; do
    echo "    $ep"
done
echo "  require complete:  $EVAL_REQUIRE_COMPLETE"
echo "  expected models:   ${#EXPECTED_MODELS[@]}"
echo

# Sanity: at least one endpoint reachable
reachable=0
for ep in "${ENDPOINTS[@]}"; do
    if curl -fsS -m 2 "${ep%/v1}/api/version" >/dev/null 2>&1; then
        reachable=$((reachable + 1))
    fi
done
if [ "$reachable" -eq 0 ]; then
    echo "no judge endpoint is reachable. start ollama first:" >&2
    echo "  bash $EVAL_DIR/scripts/start_ollama_cluster.sh start $EVAL_JUDGE_MODEL $EVAL_NUM_GPUS" >&2
    exit 1
fi
echo "$reachable / ${#ENDPOINTS[@]} endpoints reachable"
echo

CMD=(
    env
    EVAL_JUDGE_API_KEY="$EVAL_JUDGE_API_KEY"
    EVAL_JUDGE_MODEL="$EVAL_JUDGE_MODEL"
    EVAL_JUDGE_TIMEOUT_SECONDS="$EVAL_JUDGE_TIMEOUT_SECONDS"
    bash "$EVAL_DIR/run_benchmark_evaluation.sh"
    --judge-mode openai
    --judge-base-urls "${ENDPOINTS[@]}"
    --workers "$EVAL_WORKERS"
    --ground-truth "$EVAL_GT_PATH"
    --predictions-root "$EVAL_PREDICTIONS_ROOT"
    --output-dir "$EVAL_OUTPUT_DIR"
)

if [ "$EVAL_REQUIRE_COMPLETE" = "1" ]; then
    CMD+=(--require-complete-videos)
fi

if [ "${#EXPECTED_MODELS[@]}" -gt 0 ]; then
    CMD+=(--expected-models "${EXPECTED_MODELS[@]}")
fi

CMD+=("$@")

echo "== running =="
printf '  %q ' "${CMD[@]}"
echo
echo

exec "${CMD[@]}"
