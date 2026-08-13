#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ppmgs-gpu}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu118}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found. The environment can still be installed, but CUDA may not be available."
else
  nvidia-smi
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env ${ENV_NAME} already exists."
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

conda install -y -c conda-forge \
  numpy pandas scikit-learn optuna \
  bed-reader openjdk=17

python -m pip install torch torchvision torchaudio --index-url "${CUDA_WHEEL_INDEX}"

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "[OK] ${ENV_NAME} is ready."
echo "Use it with:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/train_ppmgs.py --help"
