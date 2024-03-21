#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env

#run python code
python ./utils/run_training.py --cfg ddh_march24_arc