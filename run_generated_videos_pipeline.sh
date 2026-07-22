#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

VIDEOS_DIR=""
PROMPTS_FILE="$PROJECT_ROOT/data/prompts.txt"
INFER_PROMPT_FILE="$PROJECT_ROOT/data/infer_prompt_structured.txt"
GROUND_TRUTH_OUT=""
PARSED_GROUND_TRUTH_OUT=""
GT_SUMMARY_JSON=""
OUTPUT_ROOT=""
FINAL_RESULTS_MD="$PROJECT_ROOT/FINAL_RESULTS.md"
FINAL_RESULTS_CSV=""
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GPUS="0,1,2,3,4,5,6,7"
JUDGE_MODE="openai"
SKIP_GT=0
SKIP_INFER_EVAL=0

PYTHON_BIN_DEFAULT="/data_4/liuyuan/anaconda3/envs/lifebench/bin/python"
SHIM_LIB_DEFAULT="/data_4/liuyuan/anaconda3/envs/lifebench-vlm/lib/libittnotify.so"
PYTHON_BIN="${LIFEBENCH_PYTHON:-$PYTHON_BIN_DEFAULT}"
SHIM_LIB="${LIFEBENCH_SHIM_LIB:-$SHIM_LIB_DEFAULT}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --videos-dir DIR [options]

This script runs three stages for a generated video folder:
  1. extract ground truth from prompts.txt
  2. run batched inference
  3. run evaluation and render FINAL_RESULTS.md

Required:
  --videos-dir DIR              Video root directory. Supports nested .mp4 layout.

Optional:
  --prompts-file FILE           Default: $PROMPTS_FILE
  --infer-prompt-file FILE      Default: $INFER_PROMPT_FILE
  --ground-truth-out FILE       Default: <videos-dir>/gen_prompt_generated_videos_v2.txt
  --parsed-ground-truth-out FILE
                                Default: <videos-dir>/gen_prompt_generated_videos_v2.parsed.json
  --gt-summary-json FILE        Default: <videos-dir>/gen_prompt_generated_videos_v2.summary.json
  --output-root DIR             Default: $PROJECT_ROOT/outputs/generated_videos_<RUN_ID>
  --final-results-md FILE       Default: $FINAL_RESULTS_MD
  --final-results-csv FILE      Default: <output-root>/evaluation/latest/final_model_results.csv
  --run-id ID                   Default: current UTC timestamp
  --gpus IDS                    Default: $GPUS
  --judge-mode MODE             Default: $JUDGE_MODE
  --skip-gt                     Skip GT extraction
  --skip-infer-eval             Skip inference and evaluation
  --help                        Show this help

Environment overrides:
  LIFEBENCH_PYTHON              Python executable for pipeline
  LIFEBENCH_SHIM_LIB            Optional LD_PRELOAD shim library

Example:
  $(basename "$0") \\
    --videos-dir /data_4/liuyuan/lifebench/data/generated_videos \\
    --output-root /data_4/liuyuan/lifebench/outputs/manual_run_001 \\
    --final-results-md /data_4/liuyuan/lifebench/FINAL_RESULTS.md
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-dir)
      VIDEOS_DIR="$2"
      shift 2
      ;;
    --prompts-file)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --infer-prompt-file)
      INFER_PROMPT_FILE="$2"
      shift 2
      ;;
    --ground-truth-out)
      GROUND_TRUTH_OUT="$2"
      shift 2
      ;;
    --parsed-ground-truth-out)
      PARSED_GROUND_TRUTH_OUT="$2"
      shift 2
      ;;
    --gt-summary-json)
      GT_SUMMARY_JSON="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --final-results-md)
      FINAL_RESULTS_MD="$2"
      shift 2
      ;;
    --final-results-csv)
      FINAL_RESULTS_CSV="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --judge-mode)
      JUDGE_MODE="$2"
      shift 2
      ;;
    --skip-gt)
      SKIP_GT=1
      shift
      ;;
    --skip-infer-eval)
      SKIP_INFER_EVAL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$VIDEOS_DIR" ]]; then
  echo "--videos-dir is required" >&2
  usage >&2
  exit 1
fi

VIDEOS_DIR="$(realpath "$VIDEOS_DIR")"
PROMPTS_FILE="$(realpath "$PROMPTS_FILE")"
INFER_PROMPT_FILE="$(realpath "$INFER_PROMPT_FILE")"

if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "videos dir not found: $VIDEOS_DIR" >&2
  exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "prompts file not found: $PROMPTS_FILE" >&2
  exit 1
fi
if [[ ! -f "$INFER_PROMPT_FILE" ]]; then
  echo "infer prompt file not found: $INFER_PROMPT_FILE" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$GROUND_TRUTH_OUT" ]]; then
  GROUND_TRUTH_OUT="$VIDEOS_DIR/gen_prompt_generated_videos_v2.txt"
fi
if [[ -z "$PARSED_GROUND_TRUTH_OUT" ]]; then
  PARSED_GROUND_TRUTH_OUT="$VIDEOS_DIR/gen_prompt_generated_videos_v2.parsed.json"
fi
if [[ -z "$GT_SUMMARY_JSON" ]]; then
  GT_SUMMARY_JSON="$VIDEOS_DIR/gen_prompt_generated_videos_v2.summary.json"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/generated_videos_${RUN_ID}"
fi
if [[ -z "$FINAL_RESULTS_CSV" ]]; then
  FINAL_RESULTS_CSV="$OUTPUT_ROOT/evaluation/latest/final_model_results.csv"
fi

mkdir -p "$PROJECT_ROOT/.tmp" "$OUTPUT_ROOT"
printf '%s\n' "$RUN_ID" > "$OUTPUT_ROOT/run_id.txt"

export PYTHONNOUSERSITE=1
export TMPDIR="$PROJECT_ROOT/.tmp"
export TMP="$PROJECT_ROOT/.tmp"
export TEMP="$PROJECT_ROOT/.tmp"

if [[ -f "$SHIM_LIB" ]]; then
  export LD_PRELOAD="$SHIM_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "python: $PYTHON_BIN"
echo "videos_dir: $VIDEOS_DIR"
echo "output_root: $OUTPUT_ROOT"
echo "run_id: $RUN_ID"

if [[ "$SKIP_GT" -eq 0 ]]; then
  echo "[1/3] extracting ground truth"
  "$PYTHON_BIN" "$PROJECT_ROOT/prepare_generated_videos_ground_truth.py" \
    --videos-root "$VIDEOS_DIR" \
    --prompts-file "$PROMPTS_FILE" \
    --ground-truth-out "$GROUND_TRUTH_OUT" \
    --parsed-ground-truth-out "$PARSED_GROUND_TRUTH_OUT" \
    --summary-json "$GT_SUMMARY_JSON"
else
  echo "[1/3] skipping ground truth extraction"
fi

if [[ "$SKIP_INFER_EVAL" -eq 0 ]]; then
  echo "[2/3 + 3/3] running inference and evaluation"
  "$PYTHON_BIN" "$PROJECT_ROOT/evaluation/benchmark_custom_dataset.py" \
    --videos-dir "$VIDEOS_DIR" \
    --prompt-file "$INFER_PROMPT_FILE" \
    --predictions-root "$OUTPUT_ROOT/benchmark_inference" \
    --run-id "$RUN_ID" \
    --ground-truth "$GROUND_TRUTH_OUT" \
    --evaluation-output-dir "$OUTPUT_ROOT/evaluation/latest" \
    --final-results-md "$FINAL_RESULTS_MD" \
    --final-results-csv "$FINAL_RESULTS_CSV" \
    --gpus "$GPUS" \
    --resume \
    --schedule-order model-first \
    --judge-mode "$JUDGE_MODE"
else
  echo "[2/3 + 3/3] skipping inference and evaluation"
fi

echo "done"
echo "ground_truth: $GROUND_TRUTH_OUT"
echo "parsed_ground_truth: $PARSED_GROUND_TRUTH_OUT"
echo "gt_summary: $GT_SUMMARY_JSON"
echo "predictions_root: $OUTPUT_ROOT/benchmark_inference"
echo "evaluation_dir: $OUTPUT_ROOT/evaluation/latest"
echo "final_results_md: $FINAL_RESULTS_MD"
echo "final_results_csv: $FINAL_RESULTS_CSV"
