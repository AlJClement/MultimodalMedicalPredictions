#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mem-per-cpu=100G



module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

#run python code
python ./utils/run_training.py --cfg hands