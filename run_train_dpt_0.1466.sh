#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:50:00
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=short
#SBATCH --gres=gpu:v100:1

module load Anaconda3/2022.05 
source activate /data/coml-oxmedis/kebl7678/yenv

python ./utils/run_training.py --cfg oai_dpt_swin