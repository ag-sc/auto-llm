from typing import Dict, Any, Optional

import yaml
from datasets import DatasetDict

from auto_llm.dto.builder_config import DatasetSplit
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.estimator.estimator import Estimator


class TrainerFlopsEstimator(Estimator):
    """Estimate training FLOPs using the ``6 x N x D`` formula.

    Accepts either a ``config_path`` (YAML file) **or** a pre-built
    ``TrainerRunConfig`` via the *config* parameter.  When both are
    supplied, *config* takes precedence.
    """

    def __init__(
        self,
        models_meta: Dict[str, Any],
        config_path: Optional[str] = None,
        config: Optional[TrainerRunConfig] = None,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f)
            self.config = TrainerRunConfig.model_validate(raw)
        else:
            raise ValueError(
                "Either config_path or config must be provided."
            )
        self.models_meta = models_meta

    def estimate(self) -> int:
        # Training FLOPs ≈ coeff * N * D
        #   N = full model parameter count (forward & backward both flow through every weight,
        #       so LoRA does NOT reduce N — only trainable-param count is reduced)
        #   D = num_samples * avg_tokens_per_sample * num_train_epochs
        #   coeff = 6 for full fine-tuning (Kaplan et al. 2020)
        #         = 4 for LoRA / QLoRA (~2/3 of full-FT FLOPs per Thinking Machines Lab,
        #           "LoRA Without Regret", Oct 2025: 2N²+6NR vs 3N² per weight matrix)

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

        coeff = 4 if self.config.peft_config else 6
        flops = int(coeff * N * D)

        return flops

    def get_num_params(self, model_name: str) -> int:
        return int(self.models_meta[model_name].get("num_params"))
