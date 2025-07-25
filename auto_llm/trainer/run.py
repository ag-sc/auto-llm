import argparse

import yaml

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.trainer.trainer import TrainerWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    config = TrainerRunConfig.model_validate(config)
    trainer = TrainerWrapper(config=config)
    trainer.run()
