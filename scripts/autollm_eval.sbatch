#!/bin/bash
#SBATCH --job-name=autollm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,FAIL,END

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

# set the following variables
VENV_PATH="/homes/vsudhi/autollm/venv"  # absolute path
CONFIG_PATH="config_files/evaluator_configs/test.yaml"  # relative to the root of the repo


source $VENV_PATH/bin/activate
python -m autollm.evaluator.run --config_path $CONFIG_PATH

