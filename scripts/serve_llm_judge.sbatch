#!/bin/bash
#SBATCH --job-name=llm-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=logs/%x-%u-%j.out

# Serves the LLM-as-a-Judge model with vLLM (OpenAI-compatible API).
# The evaluator talks to it via LLM_JUDGE_BASE_URL (see auto_llm/evaluator/llm_judge.py):
#   sbatch scripts/serve_llm_judge.sbatch [judge_model] [port]
# vLLM lives in its own venv to keep its torch/transformers pins out of the eval venv:
#   python -m venv /vol/auto_llm/vllm-venv && /vol/auto_llm/vllm-venv/bin/pip install vllm
VLLM_VENV_PATH=/vol/auto_llm/vllm-venv
ENV_VARIABLES_PATH=/vol/auto_llm/env.sh
JUDGE_MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
PORT=${2:-8000}

source "$VLLM_VENV_PATH/bin/activate"
source "$ENV_VARIABLES_PATH"
# HF_TOKEN/HF_HOME (and optionally LLM_JUDGE_API_KEY) come from env.sh.

echo "Job $SLURM_JOBID on $(hostname); judge model=$JUDGE_MODEL"
echo "export LLM_JUDGE_BASE_URL=http://$(hostname):$PORT/v1"
echo "export LLM_JUDGE_MODEL=$JUDGE_MODEL"

API_KEY_ARGS=()
if [ -n "$LLM_JUDGE_API_KEY" ]; then
  API_KEY_ARGS=(--api-key "$LLM_JUDGE_API_KEY")
fi

vllm serve "$JUDGE_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  "${API_KEY_ARGS[@]}"
