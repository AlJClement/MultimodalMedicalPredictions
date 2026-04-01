#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:rtx8000:1

module load Anaconda3/2022.05

source activate /data/coml-oxmedis/kebl7678/yenv
#run python code
python ./utils/run_training.py --cfg ddh_arc_newsplits_0.01499_hrnet