import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from auto_llm.constants import EMISSION_COMPARISON_FILENAME

logger = logging.getLogger(__name__)

# Keys expected in the actual-emissions dict (works for both
# ``dataclasses.asdict(EmissionsData)`` and the raw CSV row).
_ACTUAL_DURATION = "duration"
_ACTUAL_EMISSIONS = "emissions"
_ACTUAL_ENERGY = "energy_consumed"
_ACTUAL_GPU_POWER = "gpu_power"
_ACTUAL_CPU_POWER = "cpu_power"
_ACTUAL_RAM_POWER = "ram_power"
_ACTUAL_GPU_MODEL = "gpu_model"


class EmissionComparator:
    """Compare estimated and actual emission data, compute deviations, and report.

    A **pure computation** class — it receives both *estimated_emissions* and
    *actual_emissions* as dicts and performs no file I/O.  Persistence (reading
    and writing JSON / CSV) is the caller's responsibility.

    After calling :meth:`compare`, use :meth:`get_wandb_metrics` to retrieve
    a ``{wandb_key: value}`` dict for logging via
    :class:`~auto_llm.profiler.wandb_energy_logger.WandbEnergyLogger`.

    Args:
        estimated_emissions: Dict produced by
            :meth:`~auto_llm.estimator.estimation_pipeline.EstimationPipeline.estimate`
            (or loaded from ``emission_estimate.json``).
        actual_emissions: Dict of actual emission data, typically obtained
            from ``EnergyProfiler.final_emissions_data``.  Accepts both the
            typed dict produced by ``dataclasses.asdict(EmissionsData)`` and
            the normalised output of :meth:`normalize_csv_row`.

    Example::

        estimate = pipeline.run()                     # or pipeline.estimate()
        actual   = profiler.final_emissions_data

        comparator = EmissionComparator(
            estimated_emissions=estimate,
            actual_emissions=actual,
        )
        result  = comparator.compare()
        metrics = comparator.get_wandb_metrics()      # pass to WandbEnergyLogger
    """

    # Map comparison dict keys → wandb summary keys.
    _WANDB_KEY_MAP = {
        "runtime_error_pct": "emissions/comparison/runtime_error_pct",
        "co2_error_pct": "emissions/comparison/co2_error_pct",
        "energy_error_pct": "emissions/comparison/energy_error_pct",
        "estimated_vs_actual_runtime_ratio": "emissions/comparison/runtime_ratio",
        "estimated_vs_actual_co2_ratio": "emissions/comparison/co2_ratio",
        "estimated_vs_actual_energy_ratio": "emissions/comparison/energy_ratio",
        "estimated_runtime_s": "emissions/comparison/estimated_runtime_s",
        "actual_runtime_s": "emissions/comparison/actual_runtime_s",
        "estimated_co2_g": "emissions/comparison/estimated_co2_g",
        "actual_co2_g": "emissions/comparison/actual_co2_g",
        "estimated_energy_kwh": "emissions/comparison/estimated_energy_kwh",
        "actual_energy_kwh": "emissions/comparison/actual_energy_kwh",
        "actual_gpu_power_w": "emissions/comparison/actual_gpu_power_w",
        "actual_cpu_power_w": "emissions/comparison/actual_cpu_power_w",
        "actual_ram_power_w": "emissions/comparison/actual_ram_power_w",
    }

    def __init__(
        self,
        estimated_emissions: Dict[str, Any],
        actual_emissions: Dict[str, Any],
    ):
        self.estimated_emissions = estimated_emissions
        self.actual_emissions = actual_emissions
        self._last_comparison: Optional[Dict[str, Any]] = None

    def compare(self) -> Optional[Dict[str, Any]]:
        """Run the comparison and return resulting metrics.

        Returns:
            A dict with the comparison metrics, or ``None`` if an error
            occurs.
        """
        try:
            if not self.estimated_emissions or not self.actual_emissions:
                logger.warning(
                    "Cannot compare — estimated or actual emissions data "
                    "is empty."
                )
                return None

            comparison = self._compute(
                self.estimated_emissions, self.actual_emissions
            )
            self._last_comparison = comparison
            self._log_comparison(comparison)

            return comparison

        except Exception as exc:
            logger.warning(
                "Emission comparison failed (best-effort): %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
        """Convert a string-valued CSV row to typed values for :meth:`compare`.

        Useful when actual emissions are read from CodeCarbon's
        ``emissions.csv`` instead of from
        ``EnergyProfiler.final_emissions_data``.
        """
        numeric_keys = {
            _ACTUAL_DURATION, _ACTUAL_EMISSIONS, _ACTUAL_ENERGY,
            _ACTUAL_GPU_POWER, _ACTUAL_CPU_POWER, _ACTUAL_RAM_POWER,
        }
        result: Dict[str, Any] = {}
        for key, value in row.items():
            if key in numeric_keys:
                try:
                    result[key] = float(value) if value else None
                except (ValueError, TypeError):
                    result[key] = None
            else:
                result[key] = value
        return result

    @staticmethod
    def _compute(
        estimate: Dict[str, Any], actual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Derive deviation metrics from *estimate* and *actual* dicts.

        Both dicts are expected to contain typed Python values (floats/ints).
        CSV rows should be normalised by :meth:`normalize_csv_row` beforehand.

        Compared metrics:

        - **Runtime** — estimated vs actual duration (seconds)
        - **CO₂** — estimated vs actual emissions (converted to grams)
        - **Energy** — estimated energy (TDP x time) vs actual energy_consumed
        - **GPU power** — estimated TDP vs actual measured mean GPU power
        """
        est_runtime = estimate.get("estimated_runtime_s", 0.0)
        est_co2_g = estimate.get("estimated_co2_g", 0.0)

        act_runtime = actual.get(_ACTUAL_DURATION) or 0.0
        # CodeCarbon reports CO₂ in **kg**; estimator uses **grams**.
        act_co2_kg = actual.get(_ACTUAL_EMISSIONS) or 0.0
        act_co2_g = act_co2_kg * 1000.0 

        # Energy comparison
        act_energy_kwh = actual.get(_ACTUAL_ENERGY) or 0.0
        est_energy_kwh = estimate.get("estimated_energy_kwh")

        # Actual measured mean power draw
        act_gpu_power_w = actual.get(_ACTUAL_GPU_POWER)
        act_cpu_power_w = actual.get(_ACTUAL_CPU_POWER)
        act_ram_power_w = actual.get(_ACTUAL_RAM_POWER)

        # ---- deviation metrics ----

        ### Runtime

        # Runtime absolute percentage error — how far off 
        # the estimate was, regardless of direction. 
        # e.g. 16.67% means the estimate was ~17% off.
        runtime_error_pct = (
            abs(est_runtime - act_runtime) / act_runtime * 100.0
            if act_runtime
            else None
        )

        # Runtime Directional ratio — 
        # tells which way it was off. 
        # < 1 = underestimate, > 1 = overestimate, 1.0 = perfect. 
        # e.g. 0.83 means the estimator predicted 83% of the actual time.
        runtime_ratio = (
            est_runtime / act_runtime if act_runtime else None
        )

        # Carbon emission absolute percentage error — how far off 
        # the estimate was, regardless of direction. 
        # e.g. 16.67% means the estimate was ~17% off.
        co2_error_pct = (
            abs(est_co2_g - act_co2_g) / act_co2_g * 100.0
            if act_co2_g
            else None
        )

        # Carbon emission Directional ratio — 
        # tells which way it was off. 
        # < 1 = underestimate, > 1 = overestimate, 1.0 = perfect. 
        # e.g. 0.83 means the estimator predicted 83% of the actual emissions.
        co2_ratio = est_co2_g / act_co2_g if act_co2_g else None

        energy_error_pct = None
        energy_ratio = None
        if est_energy_kwh is not None and act_energy_kwh:
            # Absolute error for energy in kWh. 
            # Only computed when both values are available 
            # (estimated energy requires TDP × time).
            energy_error_pct = (
                abs(est_energy_kwh - act_energy_kwh) / act_energy_kwh * 100.0
            )
            # Energy Directional ratio — 
            # tells which way it was off. 
            # < 1 = underestimate, > 1 = overestimate, 1.0 = perfect. 
            # e.g. 0.83 means the estimator predicted 83% of the actual emissions.
            energy_ratio = est_energy_kwh / act_energy_kwh

        return {
            # raw values
            "estimated_runtime_s": est_runtime,
            "actual_runtime_s": act_runtime,
            "estimated_co2_g": est_co2_g,
            "actual_co2_g": act_co2_g,
            "estimated_energy_kwh": est_energy_kwh,
            "actual_energy_kwh": act_energy_kwh,
            # deviation metrics
            "runtime_error_pct": runtime_error_pct,
            "co2_error_pct": co2_error_pct,
            "energy_error_pct": energy_error_pct,
            "estimated_vs_actual_runtime_ratio": runtime_ratio,
            "estimated_vs_actual_co2_ratio": co2_ratio,
            "estimated_vs_actual_energy_ratio": energy_ratio,
            # actual power draw (for reference)
            "actual_gpu_power_w": act_gpu_power_w,
            "actual_cpu_power_w": act_cpu_power_w,
            "actual_ram_power_w": act_ram_power_w,
            # context
            "gpu_name": estimate.get("gpu_name"),
            "actual_gpu_model": actual.get(_ACTUAL_GPU_MODEL),
        }

    @staticmethod
    def _log_comparison(comparison: Dict[str, Any]) -> None:
        runtime_err = comparison.get("runtime_error_pct")
        co2_err = comparison.get("co2_error_pct")
        logger.info(
            "Emission comparison — "
            "estimated runtime: %.2fs vs actual: %.2fs (error: %s) | "
            "estimated CO₂: %.2fg vs actual: %.2fg (error: %s)",
            comparison.get("estimated_runtime_s", 0),
            comparison.get("actual_runtime_s", 0),
            f"{runtime_err:.1f}%" if runtime_err is not None else "N/A",
            comparison.get("estimated_co2_g", 0),
            comparison.get("actual_co2_g", 0),
            f"{co2_err:.1f}%" if co2_err is not None else "N/A",
        )

    @staticmethod
    def save_comparison(
        comparison: Dict[str, Any], output_dir: str
    ) -> None:
        """Write *comparison* dict to ``emission_comparison.json``."""
        os.makedirs(output_dir, exist_ok=True)
        path = Path(output_dir) / EMISSION_COMPARISON_FILENAME
        with open(path, "w") as f:
            json.dump(comparison, f, indent=4)
        logger.info("Emission comparison saved to %s", path)

    # ------------------------------------------------------------------
    # Wandb integration
    # ------------------------------------------------------------------

    def get_wandb_metrics(self) -> Optional[Dict[str, Any]]:
        """Return last comparison metrics as a wandb-ready ``{wandb_key: value}`` dict.

        Returns ``None`` if :meth:`compare` has not been called or returned
        ``None``.  This method does **not** create a wandb run — pass the
        result to
        :meth:`~auto_llm.profiler.wandb_energy_logger.WandbEnergyLogger.log`.
        """
        if self._last_comparison is None:
            return None

        metrics: Dict[str, Any] = {}
        for src_key, wandb_key in self._WANDB_KEY_MAP.items():
            value = self._last_comparison.get(src_key)
            if value is not None:
                metrics[wandb_key] = value
        return metrics
