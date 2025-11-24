#!/bin/bash

#BSUB -J im_job
#BSUB -o logs/im_job_%J.out
#BSUB -e logs/im_job_%J.err

# Force job onto specific host
#BSUB -m "cn650!"

# Resource requirements
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -R "affinity[thread]"
#BSUB -R "span[hosts=1]"

# Load modules
ml Singularity

# Run your job inside a Singularity container
singularity exec rootproject_latest.sif python main_pipeline.py
