#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_OUTPUTS_DIR="${SCRIPT_DIR}/batch_outputs"

PYTHON_BIN="${PYTHON_BIN:-/data_4/liuyuan/anaconda3/envs/swift-3.6/bin/python}"
ENV_ROOT="${ENV_ROOT:-/data_4/liuyuan/anaconda3/envs/swift-3.6}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
WORKER_PLAN="${WORKER_PLAN:-auto}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_INFLIGHT_VIDEOS="${MAX_INFLIGHT_VIDEOS:-64}"
WEB_DATASET="${WEB_DATASET:-web}"

usage() {
    cat <<'EOF'
Usage:
  public_data_filter/run_web_persistent.sh [OUTPUT_ROOT]

Defaults:
  - creates public_data_filter/batch_outputs/web_<UTC timestamp>
  - only scans data/public_data/web
  - uses the persistent five-agent runner
  - writes console output to run.log via tee

Environment overrides:
  PYTHON_BIN=/data_4/liuyuan/anaconda3/envs/swift-3.6/bin/python
  ENV_ROOT=/data_4/liuyuan/anaconda3/envs/swift-3.6
  GPUS=0,1,2,3,4,5,6,7
  WORKER_PLAN=auto
  SEED=0
  LOG_EVERY=10
  MAX_INFLIGHT_VIDEOS=64
  WEB_DATASET=web_h264
  EXTRA_ARGS="--early-stop-majority"
  DRY_RUN=1

Examples:
  public_data_filter/run_web_persistent.sh
  public_data_filter/run_web_persistent.sh /data_4/liuyuan/lifebench/public_data_filter/batch_outputs/web_20260429T155511Z
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    OUTPUT_ROOT="$1"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-${BATCH_OUTPUTS_DIR}/web_$(date -u +%Y%m%dT%H%M%SZ)}"
fi

mkdir -p "${OUTPUT_ROOT}"

WEB_ARGS=(
    --data-root "${PROJECT_ROOT}/data/public_data"
    --datasets "${WEB_DATASET}"
    --replace-tarsier2-with-minicpmv45
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    WEB_ARGS+=(--dry-run)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    USER_EXTRA_ARGS=(${EXTRA_ARGS})
    WEB_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

export PYTHON_BIN
export ENV_ROOT
export GPUS
export WORKER_PLAN
export SEED
export LOG_EVERY
export MAX_INFLIGHT_VIDEOS
export EXTRA_ARGS="${WEB_ARGS[*]}"
export PATH="/data_4/liuyuan/anaconda3/envs/lifebench/bin:${PATH}"

cd "${PROJECT_ROOT}"

echo "====== LifeBench web persistent run ======"
echo "project root: ${PROJECT_ROOT}"
echo "python:       ${PYTHON_BIN}"
echo "env root:     ${ENV_ROOT}"
echo "output root:  ${OUTPUT_ROOT}"
echo "gpus:         ${GPUS}"
echo "worker plan:  ${WORKER_PLAN}"
echo "dataset:      ${WEB_DATASET}"
echo "seed:         ${SEED}"
echo

"${SCRIPT_DIR}/run_persistent_resume.sh" "${OUTPUT_ROOT}" 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
