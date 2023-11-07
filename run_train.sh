#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=long
#SBATCH --mem-per-cpu=380G

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env

#run python code
python ./utils/run_training.py --cfg ddh