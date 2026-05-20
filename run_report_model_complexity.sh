#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "$*" >&2
  return 1 2>/dev/null || exit 1
}

SCRIPT_PATH="${BASH_SOURCE[0]-$0}"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"
  else
    die "CONDA_ENV_PATH is set, but 'conda' is not available in this shell."
  fi
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
  else
    die "CONDA_ENV_NAME is set, but 'conda' is not available in this shell."
  fi
fi

cd "${REPO_ROOT}"

if ! python -c "import torch" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Python in the current shell does not have PyTorch installed.

Activate the project environment first, or set one of:
  CONDA_ENV_PATH=/path/to/env
  CONDA_ENV_NAME=env_name

Example:
  module load Anaconda3/2022.05
  source activate /data/coml-oxmedis/kebl7678/yenv
  ./run_report_model_complexity.sh --summary-style
EOF
  return 1 2>/dev/null || exit 1
fi

python utils/report_model_complexity.py \
  --cfgs \
    ddh_0.01466_dpt_mm_channels \
    ddh_0.01466_dpt \
    ddh_0.01466_hrnet \
  "$@"
