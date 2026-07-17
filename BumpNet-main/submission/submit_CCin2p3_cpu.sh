#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=atlas
#SBATCH --job-name=BN
#SBATCH --output=%x_%A_%a.out
#SBATCH --partition=hpc
#SBATCH --mem=16G               # memory per node
#SBATCH --time=0-24:00          # time limit (DD-HH:MM)


#SBATCH --ntasks=1              # number of tasks
#SBATCH --nodes=1               # number of nodes
#SBATCH --cpus-per-task=4       # number of CPU threads per node
#---------------------------------------------------------------------

# Usage: sbatch --job-name=[name] submit_CCin2p3_cpu.sh [config_file]

export SIF=/sps/atlas/b/bosoriopascual/BumpNet/DDP_20230324.sif

apptainer exec --nv -B /sps/atlas/s/scalvet/ML/BN/ ${SIF} python DDP.py --config $1 

