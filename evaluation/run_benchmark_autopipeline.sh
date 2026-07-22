#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$PARENT_DIR/activate_lifebench_vlm.sh" >/dev/null
exec python "$SCRIPT_DIR/benchmark_autopipeline.py" "$@"
