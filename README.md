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

#### Step 1. Task Definition
- Follow this step if you want to **add a new task**. If the task already exists, continue from Step 2. Also see the guidelines [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md).
- Create a folder under ``config_files/evaluator_configs/tasks``. See ``config_files/evaluator_configs/tasks/ad_covid_19_pico`` for example.
- Add the **Task Definition Configuration** as defined below:

```yaml
# Task Definition Configuration Template
tag:
  - pico
task: # name of the task. This is then used in the task YAMLs.

dataset_path: arrow
dataset_kwargs:
  data_files:
    test: # path of the test ds
    validation: # path of the validation ds
test_split: test
validation_split: validation

# For `doc_to_text` and `doc_to_target`, you can use the keys in the ds for prompt construction. For example: if you have a key "text", use it here as {{text}}
doc_to_text: # input to the LLM. 
doc_to_target: # expected output from the LLM.
# For further processing the results, you can define a custom function. `utils` should lie in the same path as the task YAML. `process_results` is the name of the function in `utils`
process_results: !function utils.process_results 

# define the metrics for evaluation. These metrics should be the output from the defined `utils.process_results` function.
metric_list:
  - metric: # metric name - output from `utils.process_results` function.
    aggregation: mean
    higher_is_better: true

metadata:
  version: 1.0
```

#### Step 2. Task Execution
- After defining the task, add the **Task Execution Configuration** as defined below:

```yaml
# Task Execution Configuration Template
model: hf
tasks: <task names comma separated>
model_args: pretrained=<model-path>
wandb_args: project=llm4kmu-eval,name=<run-name>
# limit: 5 # for debugging, if you want to limit the test dataset

# this is where the tasks are defined 
include_path: config_files/evaluator_configs/tasks 
```

- Running via terminal: ``python -m auto_llm.evaluator.run --config_path <config_path>``
- Running via SLURM:
  - Configure venv and config paths in the slurm script ``scripts/autollm_eval.sbatch``. 
  - Run the script: ``sbatch scripts/autollm_eval.sbatch``.