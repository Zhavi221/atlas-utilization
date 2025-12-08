#!/bin/bash
#PBS -q N
#PBS -N im_job
#PBS -o /storage/agrp/netalev/logs/test_run.out
#PBS -e /storage/agrp/netalev/logs/test_run.err
#PBS -l select=1:ncpus=4:mem=20000mb
#PBS -l io=5
#PBS 

# Move to the submission directory
cd $PBS_O_WORKDIR
# Load ATLAS environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

# Run your pipeline with index passed in
python main_pipeline.py 