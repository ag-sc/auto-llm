import argparse
import shutil
import os

import yaml
from lm_eval.__main__ import cli_evaluate

from auto_llm.evaluator.utils import parse_lm_eval_config

from auto_llm.profiler.energy_profiler import EnergyProfiler
from auto_llm.profiler.utils import parse_wandb_args
from auto_llm.estimator.estimation_pipeline import EstimationPipeline
from auto_llm.estimator.emission_comparator import EmissionComparator

# to get STDOUT in wandb. See: https://github.com/wandb/wandb/issues/2182#issuecomment-1447879531
shutil._USE_CP_SENDFILE = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Pop energy_profiling and gpu_name before forwarding config to lm-eval-harness
    energy_profiling = config.pop("energy_profiling", False)
    gpu_name = config.pop("gpu_name", None)
    
    # Allow custom dataset code (e.g. bigbio/pubmed_qa) before any task loading
    if config.get("trust_remote_code"):
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
        try:
            import datasets.config
            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        except (ImportError, AttributeError):
            pass

    # Read (but don't pop) wandb_args — lm-eval-harness still needs them.
    wandb_args = parse_wandb_args(config.get("wandb_args"))

    lm_eval_args = parse_lm_eval_config(config)

    # start LM eval harness
    if energy_profiling:
        output_dir = config.get("output_path", ".")
        wandb_project = wandb_args.get("project", "auto-llm")
        wandb_name = wandb_args.get("name", "eval")
        log_to_wandb = bool(wandb_args)

        # Pre-run: persist energy estimate (best-effort)
        EstimationPipeline(
            output_dir=output_dir,
            gpu_name=gpu_name,
            is_eval=True,
            config_path=args.config_path,
        ).run()

        with EnergyProfiler(
            output_dir=output_dir,
            project_name=wandb_project,
            experiment_name=wandb_name,
            log_to_wandb=log_to_wandb,
        ) as profiler:
            cli_evaluate(args=lm_eval_args)

        # Post-run: compare estimated vs actual (best-effort)
        EmissionComparator(
            output_dir=output_dir,
            actual_emissions=profiler.final_emissions_data,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
        ).compare()
    else:
        cli_evaluate(args=lm_eval_args)
