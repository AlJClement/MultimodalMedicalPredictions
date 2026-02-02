#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00:00
#SBATCH --clusters=htc
#SBATCH --partition=long,medium
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:rtx8000:1

module load Anaconda3/2025.06-1

source activate $DATA/yenv

#run python code
python ./utils/run_training.py --cfg oai_arc_hrnet