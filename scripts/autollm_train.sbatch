#!/bin/bash
#SBATCH --job-name=autollm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=/vol/auto_llm/logs/%x-%u-%j.out

# set the following variables
CONFIG_PATH=$1
VENV_PATH=$2
ENV_VARIABLES_PATH=$3

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH  # /homes/vsudhi/env.sh

python -m auto_llm.trainer.run --config_path $CONFIG_PATH

