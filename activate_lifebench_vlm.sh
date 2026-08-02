#!/bin/sh

if [ -n "${ENV_ROOT:-}" ] && [ -d "$ENV_ROOT" ]; then
    :
else
    ENV_ROOT=""
    for candidate in \
        "/workspace/anaconda3/envs/lifebench-vlm" \
        "/data_4/liuyuan/anaconda3/envs/lifebench-vlm" \
        "$HOME/data/caiqingyuan/env/lifebench-vlm" \
        "$HOME/miniconda3/envs/lifebench-vlm" \
        "$HOME/anaconda3/envs/lifebench-vlm" \
        "/opt/conda/envs/lifebench-vlm"
    do
        if [ -d "$candidate" ]; then
            ENV_ROOT="$candidate"
            break
        fi
    done
fi

if [ -z "$ENV_ROOT" ]; then
    echo "lifebench-vlm environment not found." >&2
    return 1 2>/dev/null || exit 1
fi

if [ -n "${BASH_SOURCE:-}" ]; then
    _ACTIVATE_SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    _ACTIVATE_SCRIPT_PATH="$0"
fi
_ACTIVATE_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$_ACTIVATE_SCRIPT_PATH")" && pwd)"
PROJECT_TMP_DIR="${LIFEBENCH_TMPDIR:-$HOME/data/caiqingyuan/env}"
SHIM_LIB="$ENV_ROOT/lib/libittnotify.so"

export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
mkdir -p "$PROJECT_TMP_DIR"
export TMPDIR="$PROJECT_TMP_DIR"
export TMP="$PROJECT_TMP_DIR"
export TEMP="$PROJECT_TMP_DIR"

if [ -f "$SHIM_LIB" ]; then
    if [ -n "${LD_PRELOAD:-}" ]; then
        export LD_PRELOAD="$SHIM_LIB:$LD_PRELOAD"
    else
        export LD_PRELOAD="$SHIM_LIB"
    fi
fi

# torchcodec needs libnvrtc.so.13 from nvidia cu13 package
_CU13_LIB="$ENV_ROOT/lib/python3.10/site-packages/nvidia/cu13/lib"
if [ -d "$_CU13_LIB" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$_CU13_LIB:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$_CU13_LIB"
    fi
fi

echo "lifebench-vlm ready"
echo "env: $ENV_ROOT"
echo "python: $(command -v python)"
echo "tmpdir: $TMPDIR"
