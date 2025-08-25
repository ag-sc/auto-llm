import argparse

import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.logger import setup_logging
from auto_llm.trainer.sft_trainer_wrapper import SftTrainerWrapper

accelerator = Accelerator()
logger = get_logger(__name__)

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Starting training with configuration from path: {args.config_path}")

    config = TrainerRunConfig.model_validate(config)

    if config.auto_llm_trainer_args.trainer_type == "sft":
        trainer = SftTrainerWrapper(config=config)
    else:
        raise NotImplementedError

    trainer.run()
