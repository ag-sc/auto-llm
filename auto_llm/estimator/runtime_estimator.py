import math
from typing import Union, Dict, Any

from auto_llm.estimator.estimator import Estimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator


# Model FLOPs Utilization: fraction of peak GPU TFLOPs actually achieved.
# Peak TFLOPs from datasheets is unreachable in practice — typical transformer
# fine-tuning runs at 20–50% MFU. 0.3 is a conservative central value covering
# both BF16 dense fine-tuning (~0.3–0.5) and QLoRA 4-bit (~0.15–0.25, dequant overhead).
DEFAULT_MFU = 0.3


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

        runtime = flops / (tflops * DEFAULT_MFU)
        return runtime
