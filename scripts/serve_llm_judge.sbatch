#!/bin/bash
#SBATCH --job-name=llm-judge
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-12
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x-%u-%j.out

# Serves the LLM-as-a-Judge model with vLLM (OpenAI-compatible API).
VENV_PATH=${1:-venv}
ENV_VARIABLES_PATH=${2:-/vol/auto_llm/env.sh}
PORT=${3:-8000}
JUDGE_MODEL=${4:-Qwen/Qwen3.5-27B}

source "$VENV_PATH/bin/activate"
source "$ENV_VARIABLES_PATH"

echo "Job $SLURM_JOBID on $(hostname); judge model=$JUDGE_MODEL"
echo "Evaluator YAML:"
echo "llm_judge:"
echo "  base_url: http://$(hostname):$PORT/v1"
echo "  model: $JUDGE_MODEL"

API_KEY_ARGS=()
if [ -n "$LLM_JUDGE_API_KEY" ]; then
  API_KEY_ARGS=(--api-key "$LLM_JUDGE_API_KEY")
fi

export VLLM_USE_V2_MODEL_RUNNER=0 

vllm serve "$JUDGE_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size "${SLURM_GPUS_ON_NODE:-1}" \
  --language-model-only \
  "${API_KEY_ARGS[@]}"
