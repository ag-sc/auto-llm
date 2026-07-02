<h1 align="center">
    AutoLLM
</h1>

<p align="center">
    <strong>⚙️ Train and ⚖️ Evaluate your LMs effortlessly!</strong>
</p>

<p align="center">
| <a href="https://llm4kmu.de/"><b>Website</b></a> 
| <a href="https://wandb.ai/llm4kmu/projects"><b>Reports</b></a> 
| <a href="https://www.linkedin.com/company/llm4kmu/"><b>LinkedIn</b></a>
|
</p>

---

# About 
**AutoLLM** supports you in finding the **right** open source model, architecture and training method for your application. Inspired by "Auto-ML" methods, **AutoLLM** automatically determines the optimal LLM configuration for a problem, train and evaluate different LLMs for your application. You can choose from different open-source models, training techniques and evaluation metrics.

The platform is part of the project "LLM4KMU". 


> Optimierter Einsatz von Open Source Large Language Models (LLMs) in kleinen und mittelständischen Unternehmen (KMUs). Mit Mitteln der Europäischen Union gefördert. 
> 
> **#efre #efrenrw #EUinmyRegion**

# 📢 Announcements

✅ Now supports ``SftTrainer`` with `conversational` and `non-conversational` datasets. Read more [here](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support).

✅ Now supports all benchmarks in `lm-eval-harness`. Read more [here](https://github.com/EleutherAI/lm-evaluation-harness).

# Getting Started

```shell
$python3.10
pip install -r requirements.txt
```

# Components

<details>
<summary>Trainer</summary>

### Running via terminal 
Run: ``python -m auto_llm.trainer.run --config_path <config_path>``

### Running via SLURM
- Configure venv and config paths in the slurm script ``scripts/autollm_train.sbatch``.
- Run the script: ``sbatch scripts/autollm_train.sbatch``.

</details>


<details>
<summary>Evaluator</summary>

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

#### Step 3. Refresh Pareto frontier panels (post-hoc)

Each eval run logs its own energy (`emissions/*`) and accuracy (`eval/*`)
metrics to wandb, **but not** the `pareto/<label>/*` fields used by the
Pareto-frontier workspace. Pareto optimality is a **cross-run** property: a
run can only know its rank relative to every other run in the project, so
the frontier must be computed once per batch — not during a single run.

After a sweep or batch of eval jobs has finished, refresh the frontier flags
and workspace panels by running the standalone backfill script. This is a
lightweight wandb-API call (no GPU, no dataset loading) — run it directly on
the cluster login/head node, inside the same venv used for eval. **Do not
submit it via `sbatch`.**

```shell
# On the cluster login node:
source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH       # exports WANDB_API_KEY

python scripts/wandb_pareto_plot.py \
    --entity <wandb-entity> \
    --project <wandb-project>
```

The script is idempotent — re-running it after new eval jobs simply
recomputes the frontier and overwrites the `pareto/<label>/*` summary fields
on every run. Useful flags:

- ``--dry-run``: report per-label frontier sizes without writing anything.
- ``--skip-panel``: only backfill summary fields, skip the workspace upsert.
- ``--skip-backfill``: only (re)create the workspace view.
- ``--tag <tag>``: restrict to runs carrying a given tag (default: ``energy-profiling``).

##### Auto-refresh option (opt-in)

If you want each energy-eval run to update the workspace itself — without
having to remember to launch `scripts/wandb_pareto_plot.py` after every batch
— add an ``auto_pareto`` block to your evaluator YAML. When the eval job
finishes and the energy metrics have been flushed to wandb, the same Slurm
job calls `refresh_pareto_workspace(...)` in-process, on the same venv,
using the same ``WANDB_API_KEY`` already exported for energy logging.

The hook lives in `auto_llm/evaluator/run.py` (right after
`WandbEnergyLogger.flush()`). It only fires when ``energy_profiling: true``
**and** ``auto_pareto.enabled: true`` are both set — so existing configs that
omit the block keep their previous behavior (no auto-refresh).

```yaml
# config_files/evaluator_configs/.../my-eval.yaml
model: hf
tasks: medmcqa,medqa_4options,...
model_args: pretrained=google/gemma-2-2b-it
wandb_args: project=open-medical-llm-energy,name=my-run
energy_profiling: true

auto_pareto:
  enabled: true
  entity: llm4kmu                       # required if WANDB_ENTITY is unset
  project: open-medical-llm-energy      # defaults to wandb_args.project
  # Everything below is optional — defaults match scripts/wandb_pareto_plot.py
  energy_key: emissions/actual_energy_consumed_kWh
  score_scale: 100.0
  tag: energy-profiling
  workspace_name: Pareto Frontier
  skip_preset: false                    # set true once preset is registered
  skip_panel: false
  skip_backfill: false
  dry_run: false
```

What the hook does, step by step:

1. After ``wandb_logger.flush()`` finishes the energy run, reads the
   ``auto_pareto`` block.
2. Resolves the wandb entity (``auto_pareto.entity`` →
   ``wandb_args.entity`` → ``WANDB_ENTITY`` env var). Aborts the refresh
   (with a warning, not an error) if none is set.
3. Calls ``refresh_pareto_workspace(...)`` from
   ``auto_llm.evaluator.plots.wandb_pareto_plot`` — the same library
   function the manual CLI script wraps, so behavior is identical.
4. The function fetches every run in the target project via the wandb
   API, recomputes the frontier for all 13 labels (1 overall + 3 groups
   + 9 tasks), writes ``pareto/<label>/*`` keys back into each run's
   summary, then upserts the saved workspace view.
5. On success, logs the workspace URL. On any failure (network glitch,
   wandb 5xx, missing entity, …), logs a single warning of the form
   ``auto_pareto refresh failed (best-effort, eval job will succeed): …``
   and the eval job still exits 0.

Trade-offs to be aware of:

- **Best-effort only.** A refresh failure never fails the eval job. If
  reliability matters for a specific batch, leave ``enabled: false`` and
  run the manual command from the login node at the end.
- **Concurrent jobs race.** When two energy-eval Slurm jobs finish at
  the same time and both attempt to refresh, both write
  ``pareto/<label>/*`` into every run's summary. Last writer wins;
  flags converge as more jobs finish. This is acceptable because the
  Pareto frontier is a deterministic function of the data already in
  wandb — every refresh that runs after the last ``wandb.finish()``
  produces the same answer.
- **Cost.** The refresh fetches every run in the project via the wandb
  API (no GPU, no dataset loading). For projects with hundreds of runs
  this adds a few seconds per eval job. On large sweeps, prefer
  ``enabled: false`` plus one manual run at the end.
- **Workspace only.** This refresh updates the live ``Pareto Frontier``
  workspace via ``wandb_workspaces.workspaces``. It does **not**
  publish a frozen Report — for that, run the manual command and call
  the ``wandb_workspaces.reports.v2`` API yourself.

When to leave ``auto_pareto`` disabled (the default):

- Local debugging — no need to overwrite the project workspace from
  every smoke test.
- Configs not running with ``energy_profiling: true`` — the hook is
  only reached when the energy block executes.
- Massive sweeps where you only care about the final frontier — let
  the sweep complete and run the manual command once on the login
  node.
</details>

# Contact Us

For software related issues and requests, please create an issue [here](https://github.com/ag-sc/auto-llm/issues).

For other questions and collaborations, please feel free to reach out to us [here](https://llm4kmu.de/).