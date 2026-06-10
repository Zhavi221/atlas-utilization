#!/bin/bash
# ===========================================================================
# Submit MC pipeline (parse_mc: true)
# Run after submit_data.sh completes (or independently)
# ===========================================================================

set -e

NUM_JOBS=12
CONFIG="configWmaxTotal_up4j_minEvt100_subleading.yaml"
CPUS_PER_JOB=4
MEM_PER_JOB="180gb"
WALLTIME="72:00:00"
SCAN_WALLTIME="04:00:00"
HIST_WALLTIME="24:00:00"
MERGE_WALLTIME="04:00:00"
QUEUE="N"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

read RUN_NAME BASE_DIR <<< $(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
m = c.get('run_metadata', {})
print(m.get('run_name', 'pipeline_run'), m.get('base_output_dir', './output'))")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_mc_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
cp "${PIPELINE_DIR}/submit_mc.sh" "${LOG_DIR}/"

echo "=============================================="
echo "ATLAS Pipeline — MC submission"
echo "Run dir: ${RUN_DIR}"
echo "=============================================="

# Stage 1: parsing + mass_calc + post_processing (MC only)
ARRAY_JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_mc_batch
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}
#PBS -J 1-${NUM_JOBS}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --tasks parsing,mass_calculating,post_processing \
    --parse-mc true \
    --batch-job-index \${PBS_ARRAY_INDEX} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.out" \
    2> "${LOG_DIR}/batch_\${PBS_ARRAY_INDEX}.err"
EOF
)
echo "Submitted MC batch array: ${ARRAY_JOB_ID}"

# Stages 2-4 identical to data script
SCAN_JOB_ID=$(qsub -W depend=afterok:"${ARRAY_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_mc_scan
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=2:mem=32gb
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

HIST_JOB_ID=$(qsub -W depend=afterok:"${SCAN_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_mc_hist
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${HIST_WALLTIME}
#PBS -J 1-${NUM_JOBS}

cd ${PIPELINE_DIR}
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py --config "${CONFIG}" \
    --tasks histogram_creation \
    --batch-job-index \${PBS_ARRAY_INDEX} \
    --total-batch-jobs ${NUM_JOBS} \
    --run-dir "${RUN_DIR}" \
    > "${LOG_DIR}/hist_\${PBS_ARRAY_INDEX}.out" \
    2> "${LOG_DIR}/hist_\${PBS_ARRAY_INDEX}.err"
EOF
)

MERGE_JOB_ID=$(qsub -W depend=afterok:"${HIST_JOB_ID}" <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_mc_merge
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

echo ""
echo "Job chain:"
echo "  Batch:      ${ARRAY_JOB_ID}"
echo "  Scan:       ${SCAN_JOB_ID}"
echo "  Histograms: ${HIST_JOB_ID}"
echo "  Merge:      ${MERGE_JOB_ID}"
echo ""
echo "MC run dir: ${RUN_DIR}"