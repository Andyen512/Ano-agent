#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_OUTPUTS_DIR="${SCRIPT_DIR}/batch_outputs"

PYTHON_BIN="${PYTHON_BIN:-/data_4/liuyuan/anaconda3/envs/lifebench-vlm/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
WORKER_PLAN="${WORKER_PLAN:-auto}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
MAX_INFLIGHT_VIDEOS="${MAX_INFLIGHT_VIDEOS:-64}"

usage() {
    cat <<'EOF'
Usage:
  public_data_filter/run_persistent_resume.sh [OUTPUT_ROOT]

Defaults:
  - resumes the newest public_data_filter/batch_outputs/* directory
  - uses persistent 8-GPU worker pool
  - enables --resume, --reuse-agent-outputs, and --keep-going

Environment overrides:
  PYTHON_BIN=/data_4/liuyuan/anaconda3/envs/lifebench-vlm/bin/python
  GPUS=0,1,2,3,4,5,6,7
  WORKER_PLAN=auto
  SEED=0
  LOG_EVERY=10
  MAX_INFLIGHT_VIDEOS=64
  OUTPUT_ROOT=/path/to/output_root
  EXTRA_ARGS="--early-stop-majority"

Examples:
  public_data_filter/run_persistent_resume.sh
  public_data_filter/run_persistent_resume.sh /data_4/liuyuan/lifebench/public_data_filter/batch_outputs/20260424T064251Z
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

find_latest_output_root() {
    if [[ ! -d "${BATCH_OUTPUTS_DIR}" ]]; then
        return 1
    fi
    find "${BATCH_OUTPUTS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-}}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="$(find_latest_output_root || true)"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${BATCH_OUTPUTS_DIR}/persistent_full"
fi

mkdir -p "${OUTPUT_ROOT}"

echo "====== LifeBench persistent resume ======"
echo "project root: ${PROJECT_ROOT}"
echo "python:       ${PYTHON_BIN}"
echo "output root:  ${OUTPUT_ROOT}"
echo "gpus:         ${GPUS}"
echo "worker plan:  ${WORKER_PLAN}"
echo "seed:         ${SEED}"
echo

cd "${PROJECT_ROOT}"

exec "${PYTHON_BIN}" \
    public_data_filter/persistent_batch_multi_view_risk_review.py \
    --output-root "${OUTPUT_ROOT}" \
    --gpus "${GPUS}" \
    --worker-plan "${WORKER_PLAN}" \
    --seed "${SEED}" \
    --resume \
    --reuse-agent-outputs \
    --keep-going \
    --max-inflight-videos "${MAX_INFLIGHT_VIDEOS}" \
    --log-every "${LOG_EVERY}" \
    ${EXTRA_ARGS:-}
