#!/bin/bash
NUM_JOBS=$(
    python calculate_jobs_amount.py --time_for_file_sec 300 --walltime_per_job_sec 86400)

echo "Submitting $NUM_JOBS array jobs"

cat <<EOF > dynamic_job.pbs
#!/bin/bash
#PBS -q N
#PBS -N parse_job_num_\$PBS_ARRAY_INDEX
#PBS -o logs/parse_job_num_\$PBS_ARRAY_INDEX.out
#PBS -e logs/parse_job_num_\$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -l walltime=24:00:00
#PBS -J 1-$NUM_JOBS

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running job index: \$PBS_ARRAY_INDEX"

python main_pipeline.py --job-index \$PBS_ARRAY_INDEX
EOF

# Submit it
qsub dynamic_job.pbs
