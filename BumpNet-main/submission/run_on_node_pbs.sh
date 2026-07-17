#PBS -m n

source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/conda/24.5.0u/envs/common/lib
export R_HOME="/usr/wipp/conda/24.5.0/lib/R"

cd ${CWD}

python DDP.py --config ${CONFIG}