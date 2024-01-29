#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env

#run python code
python ./utils/run_training.py --cfg ddh_190