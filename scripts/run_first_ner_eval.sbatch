#!/bin/bash
#SBATCH --job-name=ner-eval-cofun-zs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=/homes/mbrinner/auto-llm/logs/%x-%u-%j.out

# --- settings -------------------------------------------------------------
REPO_DIR=/homes/mbrinner/auto-llm
VENV_PATH=/vol/auto_llm/venv
ENV_VARIABLES_PATH=/vol/auto_llm/env.sh
# The first eval: COFun, zero-shot, Qwen3.5-9B, chat-template formatting.
CONFIG_PATH=$REPO_DIR/config_files/evaluator_configs/sequence_to_structured_output/ner/pre_zero_shot_cofun_ner_qwen-3.5-9b.yaml
# --------------------------------------------------------------------------

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

export LOGLEVEL=DEBUG
cd "$REPO_DIR" || exit 1
source "$VENV_PATH/bin/activate"
source "$ENV_VARIABLES_PATH"

# W&B auth: handled via ~/.netrc (`wandb login`), so no key needed here.

python -m auto_llm.evaluator.run --config_path "$CONFIG_PATH"
