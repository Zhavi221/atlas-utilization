#!/bin/bash
# ===========================================================================
# PBS Job Submission Script for ATLAS Pipeline
#
# Modes:
#   Single-job  (NUM_JOBS=1) — one PBS job runs the full pipeline.
#   Multi-job   (NUM_JOBS>1) — PBS array splits files across N jobs,
#                              then a dependent merge/histogram job runs.
#
# Stage input overrides:
#   Set MASS_CALC_INPUT, POST_PROC_INPUT, or HISTOGRAM_INPUT to point a
#   stage at an external directory instead of the run dir's default subdirs.
#
# Usage:
#   bash submit.sh
#   NUM_JOBS=4 BATCH_TASKS=mass_calculating,post_processing bash submit.sh
#   HISTOGRAM_INPUT=/storage/.../combined_processed bash submit.sh
# ===========================================================================

set -e

# --- Configuration (edit these or override via environment) ----------------
NUM_JOBS=${NUM_JOBS:-1}
CONFIG=${CONFIG:-"config.yaml"}
CPUS_PER_JOB=${CPUS_PER_JOB:-4}
MEM_PER_JOB=${MEM_PER_JOB:-"20gb"}
WALLTIME=${WALLTIME:-"25:00:00"}
MERGE_WALLTIME=${MERGE_WALLTIME:-"01:00:00"}
QUEUE=${QUEUE:-"N"}

# Task overrides (comma-separated, empty = use config as-is)
BATCH_TASKS=${BATCH_TASKS:-""}
MERGE_TASKS=${MERGE_TASKS:-""}

# Per-stage input directory overrides (empty = use run dir defaults)
MASS_CALC_INPUT=${MASS_CALC_INPUT:-""}
POST_PROC_INPUT=${POST_PROC_INPUT:-""}
HISTOGRAM_INPUT=${HISTOGRAM_INPUT:-""}

# Post-run copy: if set, copies the first histogram ROOT file here
COPY_HISTOGRAM_TO=${COPY_HISTOGRAM_TO:-""}

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Read run_name and base_output_dir from config (single source of truth) ---
read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

BASE_DIR=${BASE_DIR_OVERRIDE:-"${BASE_DIR}"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Build --stage-input flags from env vars
STAGE_INPUT_FLAGS=""
[ -n "${MASS_CALC_INPUT}" ]  && STAGE_INPUT_FLAGS+=" --stage-input mass_calculating:${MASS_CALC_INPUT}"
[ -n "${POST_PROC_INPUT}" ]  && STAGE_INPUT_FLAGS+=" --stage-input post_processing:${POST_PROC_INPUT}"
[ -n "${HISTOGRAM_INPUT}" ]  && STAGE_INPUT_FLAGS+=" --stage-input histogram_creation:${HISTOGRAM_INPUT}"

# Save config and submit script into run dir for reproducibility
cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
cp "${PIPELINE_DIR}/submit.sh" "${LOG_DIR}/"

echo "=============================================="
echo "ATLAS Pipeline Submission"
echo "=============================================="
echo "Config:       ${CONFIG}"
echo "Run dir:      ${RUN_DIR}"
echo "Num jobs:     ${NUM_JOBS}"
echo "CPUs/job:     ${CPUS_PER_JOB}"
echo "Memory/job:   ${MEM_PER_JOB}"
echo "Walltime:     ${WALLTIME}"
echo "Merge wall:   ${MERGE_WALLTIME}"
[ -n "${BATCH_TASKS}" ]     && echo "Batch tasks:  ${BATCH_TASKS}"
[ -n "${MERGE_TASKS}" ]     && echo "Merge tasks:  ${MERGE_TASKS}"
[ -n "${MASS_CALC_INPUT}" ] && echo "Mass calc in: ${MASS_CALC_INPUT}"
[ -n "${POST_PROC_INPUT}" ] && echo "Post proc in: ${POST_PROC_INPUT}"
[ -n "${HISTOGRAM_INPUT}" ] && echo "Histogram in: ${HISTOGRAM_INPUT}"
echo "=============================================="

if [ "$NUM_JOBS" -eq 1 ]; then
    # =======================================================================
    # SINGLE JOB MODE
    # =======================================================================
    echo "Submitting single job..."

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

echo "Running single pipeline job: \$(date)"
echo "Run directory: ${RUN_DIR}"

TASK_FLAG=""
[ -n "${BATCH_TASKS}" ] && TASK_FLAG="--tasks ${BATCH_TASKS}"

python -u main.py --config "${CONFIG}" \${TASK_FLAG} \
    --run-dir "${RUN_DIR}" ${STAGE_INPUT_FLAGS} \
    > "${LOG_DIR}/pipeline.out" \
    2> "${LOG_DIR}/pipeline.err"

echo "Single job finished: \$(date)"
EOF
    )
    echo "Submitted single job: ${MAIN_JOB_ID}"

else
    # =======================================================================
    # MULTI-JOB ARRAY MODE
    # =======================================================================
    echo "Submitting ${NUM_JOBS} array jobs..."

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

echo "Batch job \${PBS_ARRAY_INDEX}/${NUM_JOBS} started: \$(date)"
echo "Run directory: ${RUN_DIR}"

TASK_FLAG=""
[ -n "${BATCH_TASKS}" ] && TASK_FLAG="--tasks ${BATCH_TASKS}"

python -u main.py --config "${CONFIG}" \${TASK_FLAG} \
    --batch-job-index \${PBS_ARRAY_INDEX} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" ${STAGE_INPUT_FLAGS} \
    > "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.out" \
    2> "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.err"

echo "Batch job \${PBS_ARRAY_INDEX} finished: \$(date)"
EOF
    )

    echo "Submitted array job: ${ARRAY_JOB_ID}"

    # --- Dependent merge/histogram job -------------------------------------
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

echo "Merge/histogram job started: \$(date)"
echo "Run directory: ${RUN_DIR}"

TASK_FLAG=""
[ -n "${MERGE_TASKS}" ] && TASK_FLAG="--tasks ${MERGE_TASKS}"

if [ -n "\${TASK_FLAG}" ]; then
    python -u main.py --config "${CONFIG}" \${TASK_FLAG} \
        --run-dir "${RUN_DIR}" ${STAGE_INPUT_FLAGS} \
        > "${LOG_DIR}/merge.out" \
        2> "${LOG_DIR}/merge.err"
else
    python -u main.py --config "${CONFIG}" \
        --merge-only \
        --run-dir "${RUN_DIR}" \
        > "${LOG_DIR}/merge.out" \
        2> "${LOG_DIR}/merge.err"
fi

if [ -n "${COPY_HISTOGRAM_TO}" ]; then
    HIST_FILE=\$(ls "${RUN_DIR}/histograms/"*.root 2>/dev/null | head -1)
    if [ -n "\${HIST_FILE}" ]; then
        mkdir -p "${COPY_HISTOGRAM_TO}"
        cp "\${HIST_FILE}" "${COPY_HISTOGRAM_TO}/"
        echo "Copied \$(basename "\${HIST_FILE}") to ${COPY_HISTOGRAM_TO}/"
    fi
fi

echo "Merge/histogram job finished: \$(date)"
EOF
    )

    echo "Submitted merge job: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
fi

echo ""
echo "Monitor:  qstat -u \$USER"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${RUN_DIR}/"
