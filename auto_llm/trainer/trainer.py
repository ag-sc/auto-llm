from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.trainer.sft_trainer import SftTrainerWrapper


class TrainerWrapper:
    def __init__(self, config: TrainerRunConfig):
        self.config = config

    def run(self):
        if self.config.auto_llm_trainer_args.trainer_type == "sft":
            trainer = SftTrainerWrapper(config=self.config)
        else:
            raise NotImplementedError

        trainer.run()
