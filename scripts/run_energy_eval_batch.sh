#!/bin/bash
# Submit one sbatch job per evaluator config.
#
# Configs source can be either:
#   - a directory: every *.yaml inside is submitted, or
#   - a file:      each non-empty, non-comment line is treated as a path to a
#                  config file (relative paths resolved against CWD).
#
# The Slurm job name is overridden to match the config file basename
# (without extension), so the log files written to
# /vol/auto_llm/logs/%x-%u-%j.out are easy to map back to the config.
#
# Usage:
#   scripts/run_energy_eval_batch.sh <venv_path> <env_variables_path> \
#       [configs_source] [sbatch_script]
#
# Defaults:
#   configs_source = config_files/evaluator_configs/open-medical-llm-benchmark/energy
#   sbatch_script  = scripts/autollm_eval_study.sbatch
#
# Examples:
#   # All yaml configs in a directory (default behavior):
#   scripts/run_energy_eval_batch.sh /vol/auto_llm/venv ./env.sh
#
#   # Explicit directory + custom sbatch:
#   scripts/run_energy_eval_batch.sh /vol/auto_llm/venv ./env.sh \
#       config_files/evaluator_configs/open-medical-llm-benchmark/energy \
#       scripts/autollm_eval.sbatch
#
#   # List file containing one config path per line:
#   scripts/run_energy_eval_batch.sh /vol/auto_llm/venv ./env.sh \
#       config_files/evaluator_configs/energy_subset.txt

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <venv_path> <env_variables_path> [configs_source] [sbatch_script]" >&2
    exit 1
fi

VENV_PATH=$1
ENV_VARIABLES_PATH=$2
CONFIGS_SOURCE=${3:-config_files/evaluator_configs/open-medical-llm-benchmark/energy}
SBATCH_SCRIPT=${4:-scripts/autollm_eval_study.sbatch}

if [[ ! -f $SBATCH_SCRIPT ]]; then
    echo "Sbatch script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

configs=()

if [[ -d $CONFIGS_SOURCE ]]; then
    shopt -s nullglob
    configs=("$CONFIGS_SOURCE"/*.yaml)
    shopt -u nullglob
    source_desc="$CONFIGS_SOURCE (directory)"
elif [[ -f $CONFIGS_SOURCE ]]; then
    while IFS= read -r line || [[ -n $line ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z $line || $line == \#* ]] && continue
        if [[ ! -f $line ]]; then
            echo "Config path not found: $line" >&2
            exit 1
        fi
        configs+=("$line")
    done < "$CONFIGS_SOURCE"
    source_desc="$CONFIGS_SOURCE (list file)"
else
    echo "Configs source not found (expected directory or list file): $CONFIGS_SOURCE" >&2
    exit 1
fi

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No configs to submit from $source_desc" >&2
    exit 1
fi

echo "Submitting ${#configs[@]} jobs from $source_desc using $SBATCH_SCRIPT"

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
