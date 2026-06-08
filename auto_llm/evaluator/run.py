import argparse
import shutil

from lm_eval.config.evaluate_config import EvaluatorConfig
from auto_llm.evaluator.utils import run_lm_eval_harness

# to get STDOUT in wandb. See: https://github.com/wandb/wandb/issues/2182#issuecomment-1447879531
shutil._USE_CP_SENDFILE = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    # Load configuration from YAML
    config = EvaluatorConfig.from_config(args.config_path)

    run_lm_eval_harness(cfg=config)
