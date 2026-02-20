#!/bin/bash
# ===========================================================================
# PBS Job Submission Script for ATLAS Pipeline
#
# Single-job  (NUM_JOBS=1) — one PBS job runs the full pipeline.
# Multi-job   (NUM_JOBS>1) — PBS array splits files across N jobs,
#                            then a dependent merge job runs.
#
# All pipeline settings (tasks, paths, stage inputs) live in config.yaml.
# This script only handles PBS resource allocation and job submission.
#
# Usage:
#   Edit the variables below, then: bash submit.sh
# ===========================================================================

set -e

# --- Configuration ---------------------------------------------------------
NUM_JOBS=1
CONFIG="config.yaml"
CPUS_PER_JOB=4
MEM_PER_JOB="20gb"
WALLTIME="25:00:00"
MERGE_WALLTIME="01:00:00"
QUEUE="N"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Read run_name and base_output_dir from config -------------------------
read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
cp "${PIPELINE_DIR}/submit.sh" "${LOG_DIR}/"

echo "=============================================="
echo "ATLAS Pipeline Submission"
echo "=============================================="
echo "Config:     ${CONFIG}"
echo "Run dir:    ${RUN_DIR}"
echo "Num jobs:   ${NUM_JOBS}"
echo "CPUs/job:   ${CPUS_PER_JOB}"
echo "Mem/job:    ${MEM_PER_JOB}"
echo "Walltime:   ${WALLTIME}"
echo "=============================================="

if [ "$NUM_JOBS" -eq 1 ]; then

    MAIN_JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_pipeline
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

python -u main.py --config "${CONFIG}" --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/pipeline.out" \
    2> "${LOG_DIR}/pipeline.err"
EOF
    )
    echo "Submitted: ${MAIN_JOB_ID}"

else

    ARRAY_JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_batch
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}
#PBS -J 1-${NUM_JOBS}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

python -u main.py --config "${CONFIG}" \
    --batch-job-index \${PBS_ARRAY_INDEX} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.out" \
    2> "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.err"
EOF
    )

    echo "Submitted array: ${ARRAY_JOB_ID}"

    MERGE_JOB_ID=$(qsub -W depend=afterok:"${ARRAY_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_merge
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l io=5
#PBS -l walltime=${MERGE_WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

python -u main.py --config "${CONFIG}" \
    --merge-only \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/merge.out" \
    2> "${LOG_DIR}/merge.err"
EOF
    )

    echo "Submitted merge: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"

fi

echo ""
echo "Monitor:  qstat -u \$USER"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${RUN_DIR}/"
