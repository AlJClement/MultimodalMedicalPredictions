#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --mem-per-cpu=300G

module load Anaconda3
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env

#run python code
#python tools/train.py --cfg experiments/ultra_hip/ddh.yaml --images /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/images/img --annotations /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt --partition '.partition_0.15_0.15_0.7_0.00000.json' --output_path './output/oai'
python ./utils/run_training.py --cfg ddh