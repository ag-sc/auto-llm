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
        # Forward-pass FLOPs ≈ 2 * N * D (Kaplan et al. 2020, arXiv:2001.08361, §2.1).
        # Intuition: every parameter participates in ~1 multiply-accumulate per
        # token (= 2 FLOPs), so a forward pass over D tokens costs ~2*N*D.
        # Training adds a backward pass (~4*N per token), giving the familiar
        # 6*N*D; inference is forward-only, so the coefficient is 2.
        # Drops attention's O(n_ctx^2) term (negligible while d_model > n_ctx/12),
        # embeddings, biases, norms, softmax — all < a few % of total.
        #   N = num params, D = num_samples * avg_tokens_per_sample.

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

        D = num_samples * avg_tokens_per_sample

        flops = int(2 * N * D)

        return flops

    def get_num_params(self, model_name: str) -> int:
        N = self.models_meta[model_name].get("num_params")

        return int(N)
