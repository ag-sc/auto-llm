import argparse
import shutil

import yaml
from lm_eval.__main__ import cli_evaluate

from auto_llm.evaluator.utils import parse_lm_eval_config

from auto_llm.profiler.energy_profiler import EnergyProfiler
from auto_llm.profiler.utils import parse_wandb_args

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

    # Pop energy_profiling before forwarding config to lm-eval-harness
    energy_profiling = config.pop("energy_profiling", False)

    # Read (but don't pop) wandb_args — lm-eval-harness still needs them.
    wandb_args = parse_wandb_args(config.get("wandb_args"))

    lm_eval_args = parse_lm_eval_config(config)

    # start LM eval harness
    if energy_profiling:
        output_dir = config.get("output_path", ".")
        wandb_project = wandb_args.get("project", "auto-llm")
        wandb_name = wandb_args.get("name", "eval")

        with EnergyProfiler(
            output_dir=output_dir,
            project_name=wandb_project,
            experiment_name=wandb_name,
            log_to_wandb=bool(wandb_args),
        ):
            cli_evaluate(args=lm_eval_args)
    else:
        cli_evaluate(args=lm_eval_args)
