import math
from typing import Union, Dict, Any

from auto_llm.estimator.estimator import Estimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator


class RuntimeEstimator(Estimator):
    def __init__(
        self,
        flops_estimator: Union[TrainerFlopsEstimator, InferenceFlopsEstimator],
        gpu_params: Dict[str, Any],
        gpu_name: str,
    ):
        self.flops_estimator = flops_estimator
        self.gpu_params = gpu_params
        self.gpu_name = gpu_name

    def estimate(self) -> float:
        flops = self.flops_estimator.estimate()
        try:
            tflops = self.gpu_params[self.gpu_name].get("tflops") * math.pow(10, 12)
        except KeyError:
            raise Exception(f"GPU name not found!")

        runtime = flops / tflops
        return runtime
