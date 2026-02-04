from typing import Dict, Any

import yaml

from auto_llm.estimator.estimator import Estimator
from auto_llm.evaluator.utils import parse_lm_eval_config, get_lm_eval_tasks


class InferenceFlopsEstimator(Estimator):
    def __init__(self, config_path: str, models_meta: Dict[str, Any]):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = parse_lm_eval_config(config)

        self.models_meta = models_meta

    def estimate(self) -> int:
        # 6 * N * D
        # N = num params [get this from model config]
        # D = num samples [get this from ds config] * avg tokens per sample [get this from model config - max length] * num epochs

        # TODO: this fails when estimating inference FLOPs for full weights fine tuned models. Their pretrained field
        #  contains the FT model's name. This won't match any key in model keys.
        model_name = [
            x.replace("pretrained=", "")
            for x in self.config.model_args.split(",")
            if "pretrained=" in x
        ][0]
        N = self.get_num_params(model_name=model_name)

        tasks = get_lm_eval_tasks(lm_eval_args=self.config)
        num_samples = 0
        for key, value in tasks.items():
            num_samples += value.eval_docs.num_rows

        # TODO: is this how the argument is passed or used in lm-eval-harness?
        avg_tokens_per_sample = min(
            1024, self.models_meta[model_name].get("max_length")
        )

        num_train_epochs = 1  # setting to 1 since this is an evaluation run

        D = num_samples * avg_tokens_per_sample * num_train_epochs

        flops = int(6 * N * D)

        return flops

    def get_num_params(self, model_name: str) -> int:
        N = self.models_meta[model_name].get("num_params")

        return int(N)
