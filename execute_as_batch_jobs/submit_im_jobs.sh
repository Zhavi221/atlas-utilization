#!/bin/bash
# Submit batch jobs for IM pipeline calculation

NUM_JOBS=$(python execute_as_batch_jobs/calculate_im_jobs.py \
    --time_for_work_unit_sec 2 \
    --walltime_per_job_sec 86400 | tail -1)

echo "Submitting $NUM_JOBS array jobs for IM pipeline"

cat <<EOF > dynamic_im_job.pbs
#!/bin/bash
#PBS -q N
#PBS -N im_job_\$PBS_ARRAY_INDEX
#PBS -o logs/im_job_\$PBS_ARRAY_INDEX.out
#PBS -e logs/im_job_\$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=8:mem=20gb
#PBS -l io=5
#PBS -l walltime=25:00:00
#PBS -J 1-$NUM_JOBS

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running IM job index: \$PBS_ARRAY_INDEX"

python main_pipeline.py \
    --batch_job_index \$PBS_ARRAY_INDEX \
    --total_batch_jobs $NUM_JOBS \
    > "/storage/agrp/netalev/logs/im_job_\${PBS_ARRAY_INDEX}.out" \
    2> "/storage/agrp/netalev/logs/im_job_\${PBS_ARRAY_INDEX}.err"
EOF

qsub dynamic_im_job.pbs

