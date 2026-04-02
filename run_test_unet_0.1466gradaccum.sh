#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --clusters=htc
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=short,medium,long

module load Anaconda3/2022.05

source activate /data/coml-oxmedis/kebl7678/yenv
#run python code
python ./utils/run_test.py --cfg ddh_arc_newsplits_0.01466_gradacum_arc