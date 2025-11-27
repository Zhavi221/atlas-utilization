#!/bin/bash
#PBS -q N
#PBS -N im_job
#PBS -o im_job_default.out      
#PBS -e im_job_default.err      
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l io=5
#PBS -J 1-12

cd "$PBS_O_WORKDIR"
echo "Starting array job index: $PBS_ARRAY_INDEX"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

# Manual output redirection (this expands correctly)
python main_pipeline.py --test_run_index "$PBS_ARRAY_INDEX" \
    > "/storage/agrp/netalev/logs/test_runs/im_job_${PBS_ARRAY_INDEX}.out" \
    2> "/storage/agrp/netalev/logs/test_runs/im_job_${PBS_ARRAY_INDEX}.err"
