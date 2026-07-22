#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/data_4/liuyuan/anaconda3/envs/lifebench"
VLM_PREFIX="/data_4/liuyuan/anaconda3/envs/lifebench-vlm"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_BIN="${ENV_PREFIX}/bin/pip"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
BRIDGE_EXEC="${ENV_PREFIX}/bin/lifebench-bridge-exec"

echo "[`date '+%F %T'`] installing host-side compatibility packages"
"${PIP_BIN}" install \
  -i "${PIP_INDEX_URL}" \
  --disable-pip-version-check \
  --retries 10 \
  --timeout 300 \
  --ignore-installed \
  --no-deps \
  pyarrow==15.0.2 \
  datasets==2.21.0 \
  sentence-transformers==5.2.3 \
  scikit-image==0.22.0 \
  wandb==0.16.1 \
  av==14.2.0 \
  pyarrow-hotfix==0.6

echo "[`date '+%F %T'`] installing lightweight runtime dependencies"
"${PIP_BIN}" install \
  -i "${PIP_INDEX_URL}" \
  --disable-pip-version-check \
  --retries 10 \
  --timeout 300 \
  --ignore-installed \
  --no-deps \
  lazy_loader==0.4 \
  appdirs==1.4.4 \
  docker-pycreds==0.4.0 \
  GitPython==3.1.44 \
  gitdb==4.0.12 \
  smmap==5.0.2 \
  sentry-sdk \
  setproctitle \
  tifffile \
  PyWavelets

echo "[`date '+%F %T'`] verifying bridged imports"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import av; print('av ok')"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import datasets; print('datasets ok')"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import pyarrow; print('pyarrow ok')"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import sentence_transformers; print('sentence_transformers ok')"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import skimage; print('skimage ok')"
"${BRIDGE_EXEC}" "${PYTHON_BIN}" -c "import wandb; print('wandb ok')"

echo "[`date '+%F %T'`] verifying bridged CLIs"
"${ENV_PREFIX}/bin/huggingface-cli" --help >/dev/null
"${ENV_PREFIX}/bin/hf" --help >/dev/null
"${ENV_PREFIX}/bin/torchrun" --help >/dev/null
"${ENV_PREFIX}/bin/accelerate" --help >/dev/null
"${ENV_PREFIX}/bin/deepspeed" --help >/dev/null
"${ENV_PREFIX}/bin/ffmpeg" -version >/dev/null
"${ENV_PREFIX}/bin/ffprobe" -version >/dev/null

echo "[`date '+%F %T'`] lifebench host environment is ready"
echo "env: ${ENV_PREFIX}"
echo "bridge site-packages: ${VLM_PREFIX}/lib/python3.10/site-packages"
