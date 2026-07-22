#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mem-per-cpu=16G

module load Anaconda3
module load CUDA/11.8.0

cd /data/coml-oxmedis/kebl7678/repos/MultimodalMedicalPredictions || exit 1

PYTHON=/data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env/bin/python

echo "Python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import torch; print('Torch:', torch.__version__)"

"$PYTHON" ./utils/run_test.py \
    --cfg ddh_arc_newsplits_0.01499_hrnet
