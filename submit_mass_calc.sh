#!/bin/bash
# ===========================================================================
# submit_mass_calc.sh
#
# Runs mass calculation on already-parsed data, one chunk per PBS job.
# Each job processes exactly ONE parsed ROOT file, keeping memory low.
#
# Parsing must already be done. Set PARSED_DIR and RUN_DIR below.
#
# Usage:
#   bash submit_mass_calc.sh
# ===========================================================================

set -e

# --- Configuration ---------------------------------------------------------
CONFIG="config10GeVbin.yaml"
CPUS_PER_JOB=1                    # single CPU per job — no parallel processes
MEM_PER_JOB="16gb"                # one file at a time needs much less memory
WALLTIME="04:00:00"               # one parsed chunk takes ~10-20 min
QUEUE="N"
SUBMIT_DELAY=2                    # seconds between submissions (be polite)
MAX_JOBS=200                      # safety cap — won't submit more than this

# The existing run directory with parsed_data/ inside
RUN_DIR="/storage/agrp/marybo/DDP/BumpNet4AtlasOpenData/data/atlas_full_fixComb_fixCut_20260403_230628"
PARSED_DIR="${RUN_DIR}/parsed_data"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${RUN_DIR}/logs/mass_calc"
mkdir -p "${LOG_DIR}"

# --- Find parsed ROOT files -------------------------------------------------
mapfile -t ROOT_FILES < <(find "${PARSED_DIR}" -name "*.root" | sort)
TOTAL=${#ROOT_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No ROOT files found in ${PARSED_DIR}"
    exit 1
fi

if [ "$TOTAL" -gt "$MAX_JOBS" ]; then
    echo "WARNING: Found ${TOTAL} files but MAX_JOBS=${MAX_JOBS}. Capping."
    TOTAL=$MAX_JOBS
fi

echo "=============================================="
echo "Mass Calculation — one job per parsed chunk"
echo "=============================================="
echo "Config     : ${CONFIG}"
echo "Run dir    : ${RUN_DIR}"
echo "Parsed dir : ${PARSED_DIR}"
echo "Chunks     : ${TOTAL}"
echo "CPUs/job   : ${CPUS_PER_JOB}"
echo "Mem/job    : ${MEM_PER_JOB}"
echo "Walltime   : ${WALLTIME}"
echo "=============================================="
echo ""
echo "Submitting ${TOTAL} jobs..."
echo ""

JOB_IDS=()

for ((i=0; i<TOTAL; i++)); do
    ROOT_FILE="${ROOT_FILES[$i]}"
    CHUNK_NAME=$(basename "${ROOT_FILE}" .root)
    JOB_NUM=$((i + 1))

    JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N mc_${JOB_NUM}
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=31
#PBS -l walltime=${WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

source ${PIPELINE_DIR}/atlasenv/bin/activate

# Override input_dir to point at the single file's parent dir,
# and use batch index to select exactly this one file
python -u main.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --tasks mass_calculating \
    --batch-job-index ${JOB_NUM} \
    --total-batch-jobs ${TOTAL} \
    > "${LOG_DIR}/${CHUNK_NAME}.out" \
    2> "${LOG_DIR}/${CHUNK_NAME}.err"
EOF
    )

    JOB_IDS+=("${JOB_ID}")
    echo "  [${JOB_NUM}/${TOTAL}] ${CHUNK_NAME} → ${JOB_ID}"
    sleep ${SUBMIT_DELAY}
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs."
echo ""

# --- Submit post-processing + histogram job after all mass calc jobs finish -
# Build depend string: afterok:id1:id2:...
DEPEND_STR=$(IFS=:; echo "${JOB_IDS[*]}")

echo "Submitting post-processing + histogram job (depends on all above)..."

MERGE_JOB_ID=$(qsub -W depend=afterok:"${DEPEND_STR}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_postproc
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l io=5
#PBS -l walltime=12:00:00

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --tasks post_processing,histogram_creation \
    > "${LOG_DIR}/postproc.out" \
    2> "${LOG_DIR}/postproc.err"
EOF
)

echo "Post-processing job: ${MERGE_JOB_ID}"
echo ""
echo "Monitor : qstat -u \$USER"
echo "Logs    : ${LOG_DIR}/"
echo ""
echo "Each job uses ~1 CPU and ~16GB — one parsed chunk at a time."
echo "Memory should stay well under limit on each node."