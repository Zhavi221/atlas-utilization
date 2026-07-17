#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=1-00:00         # time limit (DD-HH:MM)
#SBATCH --nodes=1               # number of nodes
#SBATCH --mem=16000M            # memory per node
#SBATCH --cpus-per-task=40      # number of CPU threads per node
##SBATCH --gres=gpu:1            # number of GPU(s) per node
#SBATCH --job-name=DDP
#SBATCH --output=%x_%A_%a.out
##SBATCH --array=0
#---------------------------------------------------------------------

## Usage:
## Send separate jobs with different seeds
## sbatch ---array=1-10 submit_beluga_multinodes.sh configs/my_config_file.json
## Merge samples
## sbatch submit_beluga_multinodes.sh configs/my_config_file.json ---merge <sample1> <sample2>

SIF=/project/def-arguinj/shared/DDP_data/containers/DDP_20230324.sif

# Load modules
module load apptainer
module load httpproxy

# Identify options given
CONFIG=$1
shift
ARGS="$@"

MERGE="FALSE"
for i in "$ARGS"; do
  case $i in
    --merge*)
      # MERGE="${i#*=}"
      MERGE="${i/--merge/}"
      shift #past argument=value
      ;;
  esac
done

# GENERATE
if [[ ${MERGE} == "FALSE" ]]; then
     # Modify seed provided to generate step
     NAME=$(jq -r '.generate.name' $CONFIG)
     TMP=configs/multinodes_$NAME_$SLURM_ARRAY_TASK_ID.json
     echo $CONFIG
     jq --argjson VAR [$SLURM_ARRAY_TASK_ID] '.generate.seed = $VAR' $CONFIG > "$TMP"

     # Run using the modified config
     apptainer exec --nv -B /project/def-arguinj ${SIF} python DDP.py --config $TMP
fi

# MERGE
if [[ ${MERGE} != "FALSE" ]]; then
    echo $MERGE
    TMP=configs/tmp_${NAME}_merge.json
        # "name":"training_sample_DMfunc_seed_1_signal_widths_1.0_1.5_2.0_2.5_3.0_percent_with_signal_1.0",
        # "output_dir":"/project/def-arguinj/bruna/DDP_data",
    # "seed":123
    NAME=$(jq -r '.generate.name' $CONFIG)
    OUTPUT_DIR=$(jq -r '.generate.output_dir' $CONFIG)
    OUTPUT_FMT=$(jq -r '.generate.output_format' $CONFIG)
    MERGE_CONFIG='{"tasks": {"do_merge":true},
                   "merge": {
                   "shuffle":true,
                   "seed":123
                   }}'
    JSON=$(jq --arg name $NAME '.merge.name = $name' <<< $MERGE_CONFIG)
    JSON=$(jq --arg output_dir $OUTPUT_DIR '.merge.output_dir = $output_dir' <<< $JSON)
    JSON=$(jq --arg output_fmt $OUTPUT_FMT '.merge.output_format = $output_fmt' <<< $JSON)
    INPUT=''
    for f in $MERGE; do
        INPUT+='"'
        INPUT+=$f
        INPUT+='",'
    done
    INPUT="${INPUT::-1}" # Remove extra comma
    JSON=$(jq --argjson input [$INPUT] '.merge.input = $input' <<< $JSON)
    echo $JSON | jq > "$TMP"
    cat $TMP

    # Run using the modified config
    apptainer exec --nv -B /project/def-arguinj ${SIF} python DDP.py --config $TMP
fi
