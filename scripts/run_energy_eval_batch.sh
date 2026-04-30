#!/bin/bash
# Submit one sbatch job per evaluator config in a directory.
#
# The Slurm job name is overridden to match the config file basename
# (without extension), so the log files written to
# /vol/auto_llm/logs/%x-%u-%j.out are easy to map back to the config.
#
# Usage:
#   scripts/run_energy_eval_batch.sh <venv_path> <env_variables_path> \
#       [configs_dir] [sbatch_script]
#
# Defaults:
#   configs_dir   = config_files/evaluator_configs/open-medical-llm-benchmark/energy
#   sbatch_script = scripts/autollm_eval_study.sbatch
#
# Examples:
#   scripts/run_energy_eval_batch.sh /vol/auto_llm/venv ./env.sh
#   scripts/run_energy_eval_batch.sh /vol/auto_llm/venv ./env.sh \
#       config_files/evaluator_configs/open-medical-llm-benchmark/energy \
#       scripts/autollm_eval.sbatch

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <venv_path> <env_variables_path> [configs_dir] [sbatch_script]" >&2
    exit 1
fi

VENV_PATH=$1
ENV_VARIABLES_PATH=$2
CONFIGS_DIR=${3:-config_files/evaluator_configs/open-medical-llm-benchmark/energy}
SBATCH_SCRIPT=${4:-scripts/autollm_eval_study.sbatch}

if [[ ! -d $CONFIGS_DIR ]]; then
    echo "Configs directory not found: $CONFIGS_DIR" >&2
    exit 1
fi
if [[ ! -f $SBATCH_SCRIPT ]]; then
    echo "Sbatch script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
configs=("$CONFIGS_DIR"/*.yaml)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No .yaml configs found in $CONFIGS_DIR" >&2
    exit 1
fi

echo "Submitting ${#configs[@]} jobs from $CONFIGS_DIR using $SBATCH_SCRIPT"

for config in "${configs[@]}"; do
    job_name=$(basename "$config" .yaml)
    echo "Submitting: $job_name ($config)"
    sbatch \
        --job-name="$job_name" \
        "$SBATCH_SCRIPT" \
        "$config" \
        "$VENV_PATH" \
        "$ENV_VARIABLES_PATH"
done
