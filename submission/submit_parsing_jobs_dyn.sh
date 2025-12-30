#!/bin/bash
NUM_JOBS=$(
    python submission/calculate_jobs_amount.py --time_for_file_sec 20 --walltime_per_job_sec 259200 | tail -1)

if [ "$NUM_JOBS" -eq 1 ]; then
    echo "Submitting 1 single job (not an array)"
    JOB_TYPE="single"
else
    echo "Submitting $NUM_JOBS array jobs"
    JOB_TYPE="array"
fi

if [ "$JOB_TYPE" = "array" ]; then
    qsub <<EOF
#!/bin/bash
#PBS -q N
#PBS -o logs/parse_job_default.out
#PBS -e logs/parse_job_default.err
#PBS -l select=1:ncpus=10:mem=20gb
#PBS -l io=5
#PBS -l walltime=25:00:00
#PBS -J 1-$NUM_JOBS 

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running job index: \$PBS_ARRAY_INDEX"

python create_folder_for_run.py

python main_pipeline.py --batch_job_index \$PBS_ARRAY_INDEX --total_batch_jobs $NUM_JOBS \
    > "/storage/agrp/netalev/logs/real_job_\${PBS_ARRAY_INDEX}.out" \
    2> "/storage/agrp/netalev/logs/real_job_\${PBS_ARRAY_INDEX}.err"
EOF
else
    qsub <<EOF
#!/bin/bash
#PBS -q N
#PBS -N test_skipping_files
#PBS -o logs/parse_job_default.out
#PBS -e logs/parse_job_default.err
#PBS -l select=1:ncpus=10:mem=20gb
#PBS -l io=5
#PBS -l walltime=25:00:00

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running single job (batch_job_index=1)"

python create_folder_for_run.py

python main_pipeline.py --batch_job_index 1 --total_batch_jobs $NUM_JOBS \
    > "/storage/agrp/netalev/logs/test_skipping_files.out" \
    2> "/storage/agrp/netalev/logs/test_skipping_files.err"
EOF
fi