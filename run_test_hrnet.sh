#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --clusters=htc
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=short,medium
#SBATCH --mem-per-cpu=16G

module load Anaconda3
module load CUDA/11.8.0
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

#run python code

python ./utils/run_test.py --cfg ddh_hrnet_retuve