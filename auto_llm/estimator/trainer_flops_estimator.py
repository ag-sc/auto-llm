from typing import Dict, Any

import yaml
from datasets import DatasetDict

from auto_llm.dto.builder_config import DatasetSplit
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.estimator.estimator import Estimator


class TrainerFlopsEstimator(Estimator):
    def __init__(self, config_path: str, models_meta: Dict[str, Any]):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config: TrainerRunConfig = TrainerRunConfig.model_validate(config)
        self.models_meta = models_meta

    def estimate(self) -> int:
        # 6 * N * D
        # N = num params [get this from model config]
        # D = num samples [get this from ds config] * avg tokens per sample [get this from model config - max length] * num epochs

        model_name = self.config.auto_llm_trainer_args.model_name
        N = self.get_num_params(model_name=model_name)

        num_samples = (
            DatasetDict.load_from_disk(
                self.config.trainer_data_builder_config.dataset_dir
            )
            .get(DatasetSplit.TRAIN)
            .num_rows
        )

        avg_tokens_per_sample = (
            min(1024, self.models_meta[model_name].get("max_length"))
            if not self.config.trainer_args.max_length
            else self.config.trainer_args.max_length
        )

        num_train_epochs = self.config.trainer_args.num_train_epochs

        D = num_samples * avg_tokens_per_sample * num_train_epochs

        flops = int(6 * N * D)

        return flops

    def get_num_params(self, model_name: str) -> int:
        N = self.models_meta[model_name].get("num_params")

        if self.config.peft_config:
            # TODO: estimating here, otherwise we need to load the PEFT model and check for trainable params
            N = 0.5 / 100 * N

        return int(N)
