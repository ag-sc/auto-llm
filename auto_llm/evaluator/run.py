import argparse
import shutil
import os

import yaml
from lm_eval.__main__ import cli_evaluate

from auto_llm.evaluator.utils import parse_lm_eval_config

from auto_llm.profiler.energy_profiler import EnergyProfiler
from auto_llm.profiler.wandb_energy_logger import WandbEnergyLogger
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

    # Pop energy_profiling and profiler options before forwarding config
    # to lm-eval-harness (it doesn't recognise these keys).
    energy_profiling = config.pop("energy_profiling", False)
    gpu_name = config.pop("gpu_name", None)
    tracking_mode = config.pop("tracking_mode", "machine")
    force_cpu_power = config.pop("force_cpu_power", None)
    force_ram_power = config.pop("force_ram_power", None)
    
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
        pipeline = EstimationPipeline(
            output_dir=output_dir,
            gpu_name=gpu_name,
            is_eval=True,
            config_path=args.config_path,
        )
        estimate = pipeline.run()

        with EnergyProfiler(
            output_dir=output_dir,
            project_name=wandb_project,
            experiment_name=wandb_name,
            tracking_mode=tracking_mode,
            force_cpu_power=force_cpu_power,
            force_ram_power=force_ram_power,
        ) as profiler:
            cli_evaluate(args=lm_eval_args)

        # Post-run: compare estimated vs actual (best-effort)
        comparator = EmissionComparator(
            estimated_emissions=estimate or {},
            actual_emissions=profiler.final_emissions_data,
        )
        comparison = comparator.compare()
        if comparison:
            EmissionComparator.save_comparison(comparison, output_dir)

        # Log all energy metrics to a single wandb run
        if log_to_wandb:
            wandb_logger = WandbEnergyLogger(
                project=wandb_project,
                name=wandb_name,
            )
            wandb_logger.log(pipeline.get_wandb_metrics())
            wandb_logger.log(profiler.get_wandb_metrics())
            wandb_logger.log(comparator.get_wandb_metrics())
            wandb_logger.flush()
    else:
        cli_evaluate(args=lm_eval_args)
