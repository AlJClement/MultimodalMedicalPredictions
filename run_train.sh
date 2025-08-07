#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=25:00:00
#SBATCH --clusters=htc
#SBATCH --partition=long,medium
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:rtx8000:1

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

#run python code
python ./utils/run_training.py --cfg ddh_denoise_journalpaper_arc