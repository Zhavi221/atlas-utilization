#!/bin/bash
# ===========================================================================
# Submit DATA pipeline (parse_mc: false)
#
# Job chain:
#   batch array (parsing+mass_calculating) → postproc (single job) →
#   scan → histogram array → merge
# ===========================================================================

set -e

NUM_JOBS=40
CONFIG="configWmaxTotal_up4j_minEvt10_subleading.yaml"
CPUS_PER_JOB=4
MEM_PER_JOB="180gb"
WALLTIME="72:00:00"        # full walltime for parsing+mass_calc
POSTPROC_WALLTIME="72:00:00"  # post-processing on all 20 SQLite files
SCAN_WALLTIME="08:00:00"
HIST_WALLTIME="24:00:00"
MERGE_WALLTIME="08:00:00"
QUEUE="N"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_data_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
cp "${PIPELINE_DIR}/submit_data.sh" "${LOG_DIR}/"

echo "=============================================="
echo "ATLAS Pipeline — DATA submission"
echo "Run dir: ${RUN_DIR}"
echo "=============================================="

# Stage 1: parsing + mass_calculating only (per batch)
BATCH_JOB_IDS=""
for i in $(seq 1 $NUM_JOBS); do
    JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_data_batch_${i}
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --tasks parsing,mass_calculating \
    --parse-mc false \
    --batch-job-index ${i} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/batch_${i}.out" \
    2> "${LOG_DIR}/batch_${i}.err"
EOF
    )
    echo "Submitted batch ${i}: ${JOB_ID}"
    BATCH_JOB_IDS="${BATCH_JOB_IDS:+${BATCH_JOB_IDS}:}${JOB_ID}"
done
echo "All batch jobs: ${BATCH_JOB_IDS}"

# Stage 2: post-processing — single job reading ALL im_batch_*.sqlite files
POSTPROC_JOB_ID=$(qsub -W depend=afterok:"${BATCH_JOB_IDS}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_data_postproc
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l io=5
#PBS -l walltime=${POSTPROC_WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --tasks post_processing \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/postproc.out" \
    2> "${LOG_DIR}/postproc.err"
EOF
)
echo "Submitted postproc: ${POSTPROC_JOB_ID} (depends on all batches)"

# Stage 3: scan global ranges — depends on postproc
SCAN_JOB_ID=$(qsub -W depend=afterok:"${POSTPROC_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_data_scan
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=2:mem=64gb
#PBS -l io=5
#PBS -l walltime=${SCAN_WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --scan-only \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/scan.out" \
    2> "${LOG_DIR}/scan.err"
EOF
)
echo "Submitted scan: ${SCAN_JOB_ID} (depends on postproc)"

# Stage 4: histogram creation — per batch, depends on scan
HIST_JOB_IDS=""
for i in $(seq 1 $NUM_JOBS); do
    JOB_ID=$(qsub -W depend=afterok:"${SCAN_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_data_hist_${i}
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${HIST_WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --tasks histogram_creation \
    --batch-job-index ${i} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/hist_${i}.out" \
    2> "${LOG_DIR}/hist_${i}.err"
EOF
    )
    echo "Submitted hist ${i}: ${JOB_ID}"
    HIST_JOB_IDS="${HIST_JOB_IDS:+${HIST_JOB_IDS}:}${JOB_ID}"
done
echo "All hist jobs: ${HIST_JOB_IDS}"

# Stage 5: merge — depends on ALL histogram jobs
MERGE_JOB_ID=$(qsub -W depend=afterok:"${HIST_JOB_IDS}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_data_merge
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l io=5
#PBS -l walltime=${MERGE_WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --merge-only \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/merge.out" \
    2> "${LOG_DIR}/merge.err"
EOF
)
echo "Submitted merge: ${MERGE_JOB_ID} (depends on all hist jobs)"

echo ""
echo "Job chain:"
echo "  Batches:    ${BATCH_JOB_IDS}"
echo "  Postproc:   ${POSTPROC_JOB_ID}"
echo "  Scan:       ${SCAN_JOB_ID}"
echo "  Histograms: ${HIST_JOB_IDS}"
echo "  Merge:      ${MERGE_JOB_ID}"
echo ""
echo "Monitor:  qstat -u \$USER"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${RUN_DIR}/"