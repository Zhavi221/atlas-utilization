#!/bin/bash
# ===========================================================================
# PBS Job Submission Script for ATLAS Pipeline
#
# Modes:
#   Single-job  (NUM_JOBS=1) — one PBS job runs the full pipeline.
#   Multi-job   (NUM_JOBS>1) — PBS array splits files across N jobs,
#                              then a dependent merge/histogram job runs.
#
# Incremental mode (INCREMENTAL=true):
#   Symlinks only unprocessed ROOT files into parsed_data/, runs mass calc +
#   post-proc in batch, then the merge job symlinks existing archive arrays
#   and creates histograms on the combined data.
#
# Usage:
#   bash submit.sh                 # uses defaults below
#   CONFIG=my.yaml bash submit.sh  # override config
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
MERGE_TASKS=${MERGE_TASKS:-"histogram_creation"}

# --- Incremental mode (merge new + archive processed arrays) ---------------
# Set INCREMENTAL=true and provide ARCHIVE_* paths to only process new files.
INCREMENTAL=${INCREMENTAL:-false}
ARCHIVE_ROOT_FILES=${ARCHIVE_ROOT_FILES:-""}
ARCHIVE_PROCESSED=${ARCHIVE_PROCESSED:-""}
BUMPNET_DATA_DIR=${BUMPNET_DATA_DIR:-""}

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

# --- Incremental data preparation ------------------------------------------
if [ "$INCREMENTAL" = true ]; then
    mkdir -p "${RUN_DIR}/parsed_data" "${RUN_DIR}/im_arrays" \
             "${RUN_DIR}/im_arrays_processed" "${RUN_DIR}/histograms"

    PROCESSED_HASHES=$(ls "${ARCHIVE_PROCESSED}/" | grep -oP '2024r-pp_[a-f0-9]+' | sort -u)
    ALL_PP_FILES=$(ls "${ARCHIVE_ROOT_FILES}/" | grep '2024r-pp')

    LINKED=0
    for f in ${ALL_PP_FILES}; do
        hash=$(echo "$f" | sed 's/.root$//')
        if ! echo "${PROCESSED_HASHES}" | grep -q "^${hash}$"; then
            ln -s "${ARCHIVE_ROOT_FILES}/${f}" "${RUN_DIR}/parsed_data/"
            LINKED=$((LINKED + 1))
        fi
    done
    echo "Incremental mode: symlinked ${LINKED} unprocessed ROOT files"
fi

# Save configs into run dir for reproducibility
cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
[ -f "${PIPELINE_DIR}/${MERGE_CONFIG}" ] && cp "${PIPELINE_DIR}/${MERGE_CONFIG}" "${LOG_DIR}/"
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
echo "Incremental:  ${INCREMENTAL}"
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

echo "Running single pipeline job"
echo "Run directory: ${RUN_DIR}"

TASK_FLAG=""
[ -n "${BATCH_TASKS}" ] && TASK_FLAG="--tasks ${BATCH_TASKS}"

python -u main.py --config "${CONFIG}" \${TASK_FLAG} --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/pipeline.out" \
    2> "${LOG_DIR}/pipeline.err"
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
    --run-dir "${RUN_DIR}" \
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

if [ "${INCREMENTAL}" = true ]; then
    echo "Symlinking archive processed arrays..."
    ARCHIVE_LINKED=0
    for f in ${ARCHIVE_PROCESSED}/*_main.npy; do
        base=\$(basename "\$f")
        dst="${RUN_DIR}/im_arrays_processed/\${base}"
        if [ ! -e "\${dst}" ]; then
            ln -s "\$f" "\${dst}"
            ARCHIVE_LINKED=\$((ARCHIVE_LINKED + 1))
        fi
    done
    echo "Symlinked \${ARCHIVE_LINKED} archive _main.npy files"

    TOTAL=\$(ls "${RUN_DIR}/im_arrays_processed/"*_main.npy 2>/dev/null | wc -l)
    echo "Total processed _main.npy files: \${TOTAL}"

    python -u main.py --config "${CONFIG}" --tasks ${MERGE_TASKS} \
        --run-dir "${RUN_DIR}" \
        > "${LOG_DIR}/merge.out" \
        2> "${LOG_DIR}/merge.err"

    if [ -n "${BUMPNET_DATA_DIR}" ]; then
        HIST_FILE=\$(ls "${RUN_DIR}/histograms/"*.root 2>/dev/null | head -1)
        if [ -n "\${HIST_FILE}" ]; then
            mkdir -p "${BUMPNET_DATA_DIR}"
            cp "\${HIST_FILE}" "${BUMPNET_DATA_DIR}/"
            echo "Copied histogram to ${BUMPNET_DATA_DIR}/"
        fi
    fi
else
    python -u main.py --config "${CONFIG}" \
        --merge-only \
        --run-dir "${RUN_DIR}" \
        > "${LOG_DIR}/merge.out" \
        2> "${LOG_DIR}/merge.err"
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
