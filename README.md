<h1 align="center">
    AutoLLM
</h1>

<p align="center">
    <strong>Train and Evaluate your LMs effortlessly. 😊</strong>
</p>


**TODO:** some description of the package/project. :)


Find the Weights and Biases project here: [LLM4KMU W&B](https://wandb.ai/llm4kmu/projects).


## 📢 Announcements

✅ Now supports ``SftTrainer`` with `conversational` and `non-conversational` datasets. Read more [here](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support).

✅ Now supports all benchmarks in `lm-eval-harness`. Read more [here](https://github.com/EleutherAI/lm-evaluation-harness).

## Setup

```shell
$python3.10
pip install -r requirements.txt
```

## Trainer



### Running via terminal 
Run: ``python -m auto_llm.trainer.run --config_path <config_path>``

### Running via SLURM
- Configure venv and config paths in the slurm script ``scripts/autollm_train.sbatch``.
- Run the script: ``sbatch scripts/autollm_train.sbatch``.

## Evaluator



### Running via terminal 
Run: ``python -m auto_llm.evaluator.run --config_path <config_path>``

### Running via SLURM
- Configure venv and config paths in the slurm script ``scripts/autollm_eval.sbatch``.
- Run the script: ``sbatch scripts/autollm_eval.sbatch``.