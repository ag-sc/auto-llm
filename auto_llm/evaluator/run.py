import argparse
import shutil

import yaml
from lm_eval.__main__ import cli_evaluate

from auto_llm.evaluator.utils import parse_lm_eval_config

from auto_llm.profiler.energy_profiler import EnergyProfiler

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

    lm_eval_args = parse_lm_eval_config(config)
    # Pop energy_profiling before forwarding config to lm-eval-harness
    energy_profiling = config.pop("energy_profiling", False)

    # set values from the YAML config
    lm_eval_parser = setup_parser()
    for key, value in config.items():
        lm_eval_parser.set_defaults(**{key: value})
    lm_eval_args = lm_eval_parser.parse_args(
        args=[]
    )  # passing an empty list, otherwise sys.argv[:1] is taken by default

    # start LM eval harness
    # TODO: use `lm_eval.evaluator.simple_evaluate()` instead of `lm_eval.evaluator.cli_evaluate()`?
    # if we use simple_evaluate() we can also integrate the energy profiler logging in the wandb session of lm_eval.
    if energy_profiling:
        output_dir = config.get("output_path", ".")

        with EnergyProfiler(
            output_dir=output_dir,
        ):
            cli_evaluate(args=lm_eval_args)
    else:
        cli_evaluate(args=lm_eval_args)
