#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mem-per-cpu=16G

module load Anaconda3
module load CUDA/11.8.0

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

echo "Python: $(which python)"
python --version

python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"

python ./utils/run_test.py --cfg ddh_arc_newsplits_0.01499_hrnet
#run python code

