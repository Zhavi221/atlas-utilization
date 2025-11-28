#!/bin/bash
NUM_JOBS=$(python calculate_jobs_v2.py \
    --walltime_hours 24 \
    --cores_per_node 32 \
    --memory_per_node_gb 30 \
    --mb_per_sec 2.34 | tail -1)

echo "Submitting $NUM_JOBS array jobs"

cat <<EOF > dynamic_job.pbs
#!/bin/bash
#PBS -q N
#PBS -N parse_job_\$PBS_ARRAY_INDEX
#PBS -o logs/parse_job_\$PBS_ARRAY_INDEX.out
#PBS -e logs/parse_job_\$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=32:mem=20gb
#PBS -l io=5
#PBS -l walltime=25:00:00
#PBS -J 0-$((NUM_JOBS-1))

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running job index: \$PBS_ARRAY_INDEX"

python create_folder_for_run.py

# Single instance per job (your current approach)
python main_pipeline.py \
    --batch_job_index \$PBS_ARRAY_INDEX \
    --total_batch_jobs $NUM_JOBS
EOF

qsub dynamic_job.pbs