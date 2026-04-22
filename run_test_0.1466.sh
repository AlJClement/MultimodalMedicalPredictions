#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:v100:1

module load Anaconda3/2022.05

source activate /data/coml-oxmedis/kebl7678/yenv
#run python code
python ./utils/run_test.py --cfg ddh_0.01466_dpt_test
