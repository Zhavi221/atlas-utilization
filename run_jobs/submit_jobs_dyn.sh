#!/bin/bash

INPUT_DIR="input_root_files"
FILES_PER_JOB=5

# Get number of jobs from Python
NUM_JOBS=$(python calculate_jobs_amount.py --input_dir "$INPUT_DIR" --files_per_job "$FILES_PER_JOB")

echo "Submitting $NUM_JOBS array jobs"

# Create a temporary PBS script
cat <<EOF > dynamic_job.pbs
#!/bin/bash
#PBS -q N
#PBS -N im_job
#PBS -o im_job_\$PBS_ARRAY_INDEX.out
#PBS -e im_job_\$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=04:00:00
#PBS -J 1-$NUM_JOBS

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running job index: \$PBS_ARRAY_INDEX"

python main_pipeline.py --job-index \$PBS_ARRAY_INDEX --files-per-job $FILES_PER_JOB
EOF

# Submit it
qsub dynamic_job.pbs
