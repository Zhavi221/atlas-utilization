#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=1-00:00         # time limit (DD-HH:MM)
#SBATCH --nodes=1               # number of nodes
#SBATCH --mem=16000M            # memory per node
#SBATCH --cpus-per-task=40      # number of CPU threads per node
#SBATCH --gres=gpu:1            # number of GPU(s) per node
#SBATCH --job-name=DDP
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

SIF=/project/def-arguinj/shared/DDP_data/containers/DDP_20230324.sif

# Load modules
module load apptainer
module load httpproxy

apptainer exec --nv -B /project/def-arguinj ${SIF} python DDP.py --config $@
