#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=1-00:00         # time limit (DD-HH:MM)
#SBATCH --nodes=1               # number of nodes
#SBATCH --mem=16000M            # memory per node
#SBATCH --cpus-per-task=40      # number of CPU threads per node
#SBATCH --job-name=DDP
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#----------------------------------------------------------------------

# Load modules
module load apptainer
module load httpproxy

# Copy container to node
export INPUT_PATH=$SLURM_TMPDIR
echo "Copying smooth.sif to local node"
cp -r /project/def-arguinj/shared/DDP_data/DDP_containers/smooth.sif $INPUT_PATH

# Containers
SIF=/project/def-arguinj/shared/DDP_data/DDP_containers/DDP_20230324.sif
SIF_SMOOTH=$INPUT_PATH/smooth.sif

CONFIG=$@

# Execute smoothing step in dedicated container
DO_SMOOTH=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('tasks', {}).get('do_smooth', False))
")

if [ \"$DO_SMOOTH\" = \"True\" ]; then
    TMP_CONFIG_SMOOTH=$(mktemp /tmp/config_smooth_XXXXXX.yaml)
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for k in cfg.get('tasks', {}):
    cfg['tasks'][k] = False
cfg['tasks']['do_smooth'] = True
with open('$TMP_CONFIG_SMOOTH', 'w') as f:
    yaml.dump(cfg, f)
    "

    apptainer exec --nv -B /project/def-arguinj ${SIF_SMOOTH} bash -c "
        eval \"\$(micromamba shell hook --shell bash)\"
        micromamba activate base
        python DDP.py --config $TMP_CONFIG_SMOOTH
    "

    rm -f $TMP_CONFIG_SMOOTH
fi

# Execute all other steps in regular container
TMP_CONFIG=$(mktemp /tmp/config_tmp_XXXXXX.yaml)
python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['tasks']['do_smooth'] = False
with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
"

apptainer exec --nv -B /project/def-arguinj ${SIF} python DDP.py --config $TMP_CONFIG

rm -f $TMP_CONFIG