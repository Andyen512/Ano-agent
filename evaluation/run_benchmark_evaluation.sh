#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="python3"
if . "$PARENT_DIR/activate_lifebench_vlm.sh" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_outputs.py" "$@"
