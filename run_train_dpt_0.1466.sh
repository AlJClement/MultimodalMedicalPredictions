#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:50:00
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=short
#SBATCH --gres=gpu:v100:1

module load Anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/coml-oxmedis/kebl7678/yenv

which python
python -c "import torch; print(torch.__version__)"

python ./utils/run_training.py --cfg ddh_0.01466_dpt