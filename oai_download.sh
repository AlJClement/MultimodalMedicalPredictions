#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=100G
#SBATCH --time=7-00
#SBATCH --clusters=arc
#SBATCH --partition=long
#SBATCH -e job.%J.err
#SBATCH -o job.%J.out

downloadcmd -dp 1232950 -d /data/coml-oxmedis/datasets-old/xr/ -wt 32 --verbose -u allison_clement