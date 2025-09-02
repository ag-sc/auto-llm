from typing import Dict

from typing_extensions import Any

from auto_llm.estimator.estimator import Estimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.utils import get_model_params, get_gpu_params


class EmissionEstimator(Estimator):
    def __init__(
        self,
        runtime_estimator: RuntimeEstimator,
        gpu_params: Dict[str, Any],
        gpu_name: str,
    ):
        self.runtime_estimator = runtime_estimator
        self.gpu_params = gpu_params
        self.gpu_name = gpu_name

    def estimate(self) -> float:
        # https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/emissions
        # https://mlco2.github.io/impact/
        # co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh
        #   energy_consumption_kWh = power_consumption_kW * total_evaluation_time_hours
        #       power_consumption_kW = get this from device config
        #       total_evaluation_time_hours = get this from FLOPs and TFLOPs of hardware
        #   carbon_intensity_g_per_kWh = get this from data stats for C intensity, depends on region

        runtime = self.runtime_estimator.estimate()  # runtime in seconds
        runtime_in_h = runtime / 60 / 60
        try:
            tdp = self.gpu_params[self.gpu_name].get("tdp")
        except KeyError:
            raise Exception(f"GPU name not found!")

        energy_consumption_kWh = tdp * runtime_in_h

        # Source: https://www.nowtricity.com/country/germany/
        carbon_intensity_g_per_kWh = 321

        co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh

        return co2_emissions_g


if __name__ == "__main__":
    models_meta = get_model_params()

    config_path = "config_files/evaluator_configs/pico_ad_gemma-2-2b-sft.yaml"
    flops_estimator = InferenceFlopsEstimator(
        config_path=config_path, models_meta=models_meta
    )

    gpu_params = get_gpu_params()
    runtime_estimator = RuntimeEstimator(
        flops_estimator=flops_estimator,
        gpu_params=gpu_params,
        gpu_name="NVIDIA H100 SXM5 80GB",
    )

    emission_estimator = EmissionEstimator(
        runtime_estimator=runtime_estimator,
        gpu_params=gpu_params,
        gpu_name="NVIDIA H100 SXM5 80GB",
    )
    print(emission_estimator.estimate())
