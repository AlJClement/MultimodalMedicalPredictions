#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=short
#SBATCH --gres=gpu:v100:1

export PATH=/data/coml-oxmedis/kebl7678/yenv/bin:$PATH
module load Anaconda3
module load CUDA/11.8.0
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env


python ./utils/run_test.py --cfg ddh_arc_newsplits_0.01499_hrnet
