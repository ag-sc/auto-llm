from typing import Dict

from typing_extensions import Any

from auto_llm.estimator.estimator import Estimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator


# CPU model: 2× AMD EPYC 7713 @ 225 W each. Same value passed to CodeCarbon
# via `force_cpu_power: 450`, so estimator and measurement use identical CPU assumptions.
CPU_TDP_W = 450

# Assumed CPU utilization during training. CodeCarbon uses
#   P_cpu = TDP × (0.1 + 0.9 × (load)^3)
# (10% idle floor + cubic DVFS scaling). Across training runs in emissions.json,
# measured cpu_utilization clusters at 22–25%; 0.25 yields P_cpu ≈ 51 W vs
# measured ~54 W — within 5%.
CPU_LOAD = 0.25


def _cpu_power_w() -> float:
    return CPU_TDP_W * (0.1 + 0.9 * CPU_LOAD ** 3)


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

    def estimate_energy_kwh(self) -> float:
        """Return estimated energy consumption in kWh: GPU TDP + CPU (CodeCarbon model) × runtime."""
        runtime = self.runtime_estimator.estimate()
        runtime_in_h = runtime / 3600.0
        try:
            tdp = self.gpu_params[self.gpu_name].get("tdp")
        except KeyError:
            raise Exception("GPU name not found!")
        energy_gpu_kwh = (tdp / 1000) * runtime_in_h
        energy_cpu_kwh = (_cpu_power_w() / 1000) * runtime_in_h
        return energy_gpu_kwh + energy_cpu_kwh

    def estimate(self) -> float:
        # https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/emissions
        # https://mlco2.github.io/impact/
        # co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh
        #   energy_consumption_kWh = (P_gpu + P_cpu) × runtime_h, both in kW
        #   carbon_intensity_g_per_kWh = depends on region

        energy_consumption_kWh = self.estimate_energy_kwh()

        # TODO: take region as an argument and compute carbon_intensity_g_per_kWh based on this argument
        # Source: https://www.nowtricity.com/country/germany/
        carbon_intensity_g_per_kWh = 328  # average carbon intensity in Germany in 2025

        co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh

        return co2_emissions_g
