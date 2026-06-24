#!/bin/bash
#SBATCH --job-name=ner-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=/homes/mbrinner/auto-llm/logs/%x-%u-%j.out

# Generic NER eval runner. Pass the evaluator config path as the first argument:
#   sbatch --job-name=<name> scripts/run_ner_eval.sbatch <config_path>
REPO_DIR=/homes/mbrinner/auto-llm
VENV_PATH=/vol/auto_llm/venv
ENV_VARIABLES_PATH=/vol/auto_llm/env.sh
CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
  echo "usage: sbatch [--job-name=...] scripts/run_ner_eval.sbatch <config_path>"
  exit 1
fi

echo "Job $SLURM_JOBID on $(hostname); config=$CONFIG_PATH"

export LOGLEVEL=DEBUG
cd "$REPO_DIR" || exit 1
source "$VENV_PATH/bin/activate"
source "$ENV_VARIABLES_PATH"
# W&B auth via ~/.netrc (`wandb login`); HF_TOKEN/HF_HOME come from env.sh.

python -m auto_llm.evaluator.run --config_path "$CONFIG_PATH"
