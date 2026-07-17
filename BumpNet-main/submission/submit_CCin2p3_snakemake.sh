#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=atlas
#SBATCH --job-name=BN
#SBATCH --output=%x_%A_%a.out
#SBATCH --partition=hpc
#SBATCH --mem=16G               # memory per node
#SBATCH --time=7-00:00          # time limit (DD-HH:MM)
#SBATCH --ntasks=1              # number of tasks
#SBATCH --nodes=1               # number of nodes
#SBATCH --cpus-per-task=1       # number of CPU threads per node
#---------------------------------------------------------------------

# Usage: sbatch --job-name=BNmaster submit_CCin2p3_snakemake.sh

cd /sps/atlas/s/scalvet/ML/BN/
module load conda
conda activate snakemake
snakemake --profile BumpNet/snakemake/workflow/profiles/ccin2p3/ --snakefile BumpNet/snakemake/workflow/analysis.smk  all
echo "BN" | mail -s "BN done"  scalvet@in2p3.fr