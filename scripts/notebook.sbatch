#!/bin/bash
#SBATCH --job-name=autollm-ger-ner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%u-%j.out

# set the following variables
VENV_PATH=venv/
ENV_VARIABLES_PATH=../env.sh

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH

python -m ipykernel install --user --name=venv

fuser -k 9985/tcp
jupyter lab --port 9985


