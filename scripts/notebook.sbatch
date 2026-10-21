#!/bin/bash
#SBATCH --job-name=knowledge-conflicts
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-14
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=%x-%u-%j.out

# set the following variables
VENV_PATH=venv/
ENV_VARIABLES_PATH=env.sh

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH  # /homes/vsudhi/env.sh

python -m ipykernel install --user --name=venv

fuser -k 8081/tcp
jupyter lab --port 8081

