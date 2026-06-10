#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --clusters=htc
#SBATCH --partition=short,interactive
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:1

module load Anaconda3
module load CUDA/11.8.0
source activate /data/coml-oxmedis/kebl7678/conda_envs/mm_env2/mm_env

python ./ere_truerad.py \
  --cfg ddh_0.01499_dpt_mm_channels \
  --model_path /data/coml-oxmedis/allie/repos/MultimodalMedicalPredictions/output_ddh_0427_dptmmgate_0.1499/model:1/_model_run:1_idx.pth \
  --fit_temperature \
  --temperature_epochs 30 \
  --temperature_lr 0.001

python ./ere_truerad.py \
  --cfg 'ddh_0.01499_dpt ' \
  --model_path /data/coml-oxmedis/allie/repos/MultimodalMedicalPredictions/output_ddh_0420_dpt_0.1499/model:1/_model_run:1_idx.pth \
  --fit_temperature \
  --temperature_epochs 30 \
  --temperature_lr 0.001

python ./ere_truerad.py \
  --cfg 'ddh_0.01499_dpt ' \
  --model_path /data/coml-oxmedis/allie/repos/MultimodalMedicalPredictions/output_ddh_0420_dpt_0.1499/model:1/_model_run:1_idx.pth \
  --fit_temperature \
  --temperature_epochs 30 \
  --temperature_lr 0.001 \
  --temperature_bins 3 \
  --ece_curve_mode ere_radial_error
