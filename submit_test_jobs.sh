#!/bin/bash
#PBS -q N
#PBS -N im_job
#PBS -o im_job_$PBS_ARRAY_INDEX.out
#PBS -e im_job_$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=1:mem=1000mb
#PBS -l io=5
#PBS -J 1-10  

# Move to the submission directory
cd $PBS_O_WORKDIR

echo "Starting array job index: $PBS_ARRAY_INDEX"

# Load ATLAS environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

# Run your pipeline with index passed in
python main_pipeline.py --test_run_index $PBS_ARRAY_INDEX
