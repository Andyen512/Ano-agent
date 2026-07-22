#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec python3 "$SCRIPT_DIR/supervise_command_on_user_busy.py" \
  --watch-user caochunshui \
  --mode any \
  --poll-interval 5 \
  --busy-hold-seconds 10 \
  --idle-hold-seconds 30 \
  --pid-file "$SCRIPT_DIR/outputs/benchmark_inference/benchmark_autopipeline_supervised.pid" \
  -- \
  "$SCRIPT_DIR/evaluation/run_benchmark_autopipeline.sh" "$@"
