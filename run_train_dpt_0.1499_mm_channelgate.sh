#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --clusters=htc
#SBATCH --mem=50G
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1

module load Anaconda3/2022.05 
source activate /data/coml-oxmedis/kebl7678/yenv
#run python code
python ./utils/run_training.py --cfg ddh_0.01499_dpt_mm_channels