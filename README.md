# Auto-LLM

TODO: some description of the package/project. :)


Find the Weights and Biases project here: [LLM4KMU W&B](https://wandb.ai/llm4kmu/projects).


## Setup

```shell
$python3.10
pip install -r requirements.txt
```

## Trainer

...

## Evaluator

### Running via terminal 
Run: ``python -m auto_llm.evaluator.run --config_path <config_path>``

### Running via SLURM
- Configure the venv and config paths in the slurm script ``scripts/autollm_eval.sbatch``.
- Run the script: ``sbatch scripts/autollm_eval.sbatch``.