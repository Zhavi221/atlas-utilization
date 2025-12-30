#PBS -m n
#!/bin/bash
#PBS -q N
#PBS -l select=1:ncpus=4:mem=20gb
#PBS -l io=5
#PBS -l walltime=72:00:00

SIF=/srv01/agrp/illanb/DDP/containers/BumpNet_20250828.sif
STORAGE_MNT_DIR=/storage/agrp/netalev/testing_BumpNet
MNT_DIR=/srv01/agrp/netalev/BumpNet-main

CWD=/srv01/agrp/netalev/BumpNet-main
CONFIG=${CWD}/configs/analysis.yaml

source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate apptainer

cd ${CWD}

apptainer exec --cleanenv --nv -B ${MNT_DIR},${STORAGE_MNT_DIR} ${SIF} python DDP.py --config ${CONFIG}