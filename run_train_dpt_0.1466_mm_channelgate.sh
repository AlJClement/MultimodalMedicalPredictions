#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=11:50:00
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=50G
#SBATCH --partition=short

module load Anaconda3/2022.05 
source activate /data/coml-oxmedis/kebl7678/yenv

#run python code
python ./utils/run_training.py --cfg ddh_0.01466_dpt_mm_channels