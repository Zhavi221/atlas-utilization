#!/bin/bash
# ===========================================================================
# PBS Job Submission Script for ATLAS Pipeline
#
# Architecture:
#   - Single-job mode (NUM_JOBS=1): one PBS job runs the full pipeline
#   - Multi-job mode  (NUM_JOBS>1): PBS array where each job runs the
#     FULL pipeline (parse → IM calc → post-proc → histograms) on its
#     file slice.  A final merge job combines histograms (hadd),
#     aggregates stats, and generates plots.
#
# Usage:
#   # Edit NUM_JOBS below, then:
#   bash submission/submit.sh
#
#   # Or override on command line:
#   NUM_JOBS=4 bash submit.sh
# ===========================================================================

# --- Configuration ---
NUM_JOBS=${NUM_JOBS:-1}
CONFIG=${CONFIG:-"config.yaml"}
CPUS_PER_JOB=${CPUS_PER_JOB:-4}
MEM_PER_JOB=${MEM_PER_JOB:-20gb}
WALLTIME=${WALLTIME:-"25:00:00"}
MERGE_WALLTIME=${MERGE_WALLTIME:-"00:30:00"}   # merge is lightweight
QUEUE=${QUEUE:-"N"}

# --- Read run_name and base_output_dir from config (single source of truth) ---
read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

# Allow env-var override (e.g. BASE_DIR=/tmp bash submission/submit.sh)
BASE_DIR=${BASE_DIR_OVERRIDE:-"${BASE_DIR}"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "ATLAS Pipeline Submission"
echo "=============================================="
echo "Config:       ${CONFIG}"
echo "Base dir:     ${BASE_DIR}"
echo "Run dir:      ${RUN_DIR}"
echo "Num jobs:     ${NUM_JOBS}"
echo "CPUs/job:     ${CPUS_PER_JOB}"
echo "Memory/job:   ${MEM_PER_JOB}"
echo "Walltime:     ${WALLTIME}"
echo "Merge wall:   ${MERGE_WALLTIME}"
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
#PBS -o logs/pipeline_single.out
#PBS -e logs/pipeline_single.err
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running single pipeline job"
echo "Run directory: ${RUN_DIR}"

python main.py --config "${CONFIG}" --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/pipeline_single.out" \
    2> "${LOG_DIR}/pipeline_single.err"
EOF
    )
    echo "Submitted single job: ${MAIN_JOB_ID}"

else
    # =======================================================================
    # MULTI-JOB ARRAY MODE
    #
    # Each batch job runs the FULL pipeline on its slice of files:
    #   parse → IM calc → post-proc → histograms
    # Each batch writes:
    #   histograms/batch_N.root   (per-batch histogram file)
    #   logs/batch_N_stats.json   (per-batch statistics)
    #
    # The merge job (--merge-only) then:
    #   1. hadd batch_*.root → atlas_opendata.root
    #   2. Aggregates stats JSONs
    #   3. Generates plots
    # =======================================================================
    echo "Submitting ${NUM_JOBS} array jobs (full pipeline per batch)..."

    ARRAY_JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_batch
#PBS -o logs/batch_\${PBS_ARRAY_INDEX}.out
#PBS -e logs/batch_\${PBS_ARRAY_INDEX}.err
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}
#PBS -J 1-${NUM_JOBS}

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "Running batch job \$PBS_ARRAY_INDEX of ${NUM_JOBS} (full pipeline)"
echo "Shared run directory: ${RUN_DIR}"

python main.py --config "${CONFIG}" \
    --batch-job-index \$PBS_ARRAY_INDEX \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.out" \
    2> "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.err"
EOF
    )

    echo "Submitted array job: ${ARRAY_JOB_ID}"

    # Submit lightweight merge job
    MERGE_JOB_ID=$(qsub -W depend=afterok:"${ARRAY_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_merge
#PBS -o logs/merge.out
#PBS -e logs/merge.err
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l io=3
#PBS -l walltime=${MERGE_WALLTIME}

cd \$PBS_O_WORKDIR

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"

echo "=== Merge job ==="
echo "Merging histograms + aggregating stats + generating plots"
echo "Run directory: ${RUN_DIR}"

python main.py --config "${CONFIG}" \
    --merge-only \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/merge.out" \
    2> "${LOG_DIR}/merge.err"

echo "Merge job complete"
EOF
    )

    echo "Submitted merge job: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
fi

echo ""
echo "Monitor with: qstat -u \$USER"
echo "Output in:    ${RUN_DIR}"
