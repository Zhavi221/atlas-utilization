#!/bin/bash
#BSUB -J TestRuns[1-12]
#BSUB -o logs/testing_run_%J_%I.out
#BSUB -e logs/testing_run_%J_%I.err
# Specify the exact, single host name
#BSUB -m "cn650!" 
# Use the explicit selection string for redundancy
#BSUB -R "select[hname==cn650] rusage[mem=14GB] span[hosts=1] affinity[thread]"
#BSUB -n 4

CONFIG_INDEX=$((LSB_JOBINDEX-1))
CONFIG_NAME=$(jq -r ".[$CONFIG_INDEX].name" testing_runs.json)

ml Singularity
singularity exec rootproject_latest.sif python main_pipeline.py --config "configs/pipeline_config.yaml" --test_run_index "$((LSB_JOBINDEX))"