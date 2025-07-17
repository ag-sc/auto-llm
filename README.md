# Auto-LLM

...


``sh
pip install -r requirements.txt
``

## Trainer

...

## Evaluator

### Running via terminal 
Run: ``python -m auto_llm.evaluator.run --config_path <config_path>``

### Running via SLURM
- Configure the venv and config paths in the slurm script ``scripts/autollm_eval.sbatch``.
- Run the script: ``sbatch scripts/autollm_eval.sbatch``.