#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --clusters=htc
#SBATCH --partition=long,medium
#SBATCH --mem=200G
#SBATCH --gres=gpu:rtx8000:1

module load Anaconda3/2022.05
source activate /data/coml-oxmedis/kebl7678/yenv

# reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# debug info
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_summary(device=0, abbreviated=True))
PY

# run with reduced batch size (example arg)
python ./utils/run_training.py --cfg "oai_arc_unet_bignet" --cases '9947240-20051013-01155804-001','9750920-20051011-01055704-001','9582125-20050815-01009004-001','9943227-20051028-01176201-001','9958234-20051013-01155704-001','9964731-20050907-01091201-001' --batch-size 1 --amp
