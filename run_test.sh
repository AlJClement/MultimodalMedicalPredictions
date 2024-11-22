#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem-per-cpu=20G

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

#run python code

python ./utils/run_test.py --cfg ddh_RNOH_arc