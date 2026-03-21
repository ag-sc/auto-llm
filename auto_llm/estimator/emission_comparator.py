import json
import logging
import wandb
import os
from pathlib import Path
from typing import Any, Dict, Optional

from auto_llm.constants import (
    EMISSION_COMPARISON_FILENAME,
    EMISSION_ESTIMATE_FILENAME,
)
from auto_llm.profiler.utils import read_last_emissions_row

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
    """Load estimated and actual emission data, compute deviations, and report.

    The estimated data is read from ``emission_estimate.json`` and the actual
    data is read directly from CodeCarbon's ``emissions.csv`` — no
    intermediate JSON file is needed.

    Args:
        output_dir: Directory containing ``emission_estimate.json`` and
            ``emissions.csv``.
        actual_emissions: Optional dict of actual emission data, typically
            obtained from ``EnergyProfiler.final_emissions_data``.  When
            provided the comparator uses it directly instead of parsing the
            CSV.  Accepts both the typed dict produced by
            ``dataclasses.asdict(EmissionsData)`` and the string-valued dict
            from ``csv.DictReader``.
        log_to_wandb: When ``True`` a dedicated wandb run is created to log
            comparison metrics under the ``emissions/comparison/`` prefix.
        wandb_project: Wandb project name for the comparison run.
        wandb_name: Base name for the comparison wandb run (suffixed with
            ``-comparison``).

    
    Example::

    comparator = EmissionComparator(
        output_dir="/out/model",
        log_to_wandb=True,
        wandb_project="my-project",
        wandb_name="sft-run-1",
    )
    result = comparator.compare()
    # result is a dict with error-% metrics, or None if files are missing.
    """

    def __init__(
        self,
        output_dir: str,
        actual_emissions: Optional[Dict[str, Any]] = None,
        log_to_wandb: bool = False,
        wandb_project: str = "auto-llm",
        wandb_name: str = "run",
    ):
        self.output_dir = output_dir
        self.actual_emissions = actual_emissions
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name

    
    def compare(self) -> Optional[Dict[str, Any]]:
        """Run the comparison and return resulting metrics.

        Returns:
            A dict with the comparison metrics, or ``None`` if either input
            file is missing or an error occurs.
        """
        try:
            estimate = self._load_json(EMISSION_ESTIMATE_FILENAME)

            # Prefer the in-memory data passed by the caller; fall back to CSV.
            actual = self.actual_emissions
            if actual is None:
                csv_row = read_last_emissions_row(self.output_dir)
                if csv_row is not None:
                    actual = self._normalize_csv_row(csv_row)

            if estimate is None or actual is None:
                logger.warning(
                    "Cannot compare — emission_estimate.json or actual "
                    "emissions data missing in '%s'.",
                    self.output_dir,
                )
                return None

            comparison = self._compute(estimate, actual)
            self._save_json(comparison)
            self._log_comparison(comparison)

            if self.log_to_wandb:
                self._log_comparison_to_wandb(comparison)

            return comparison

        except Exception as exc:
            logger.warning(
                "Emission comparison failed (best-effort): %s", exc
            )
            return None

    @staticmethod
    def _normalize_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
        """Convert string-valued CSV row to typed values for ``_compute``."""
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
        CSV rows are normalised by ``_normalize_csv_row`` before reaching here.

        Compared metrics:

        - **Runtime** — estimated vs actual duration (seconds)
        - **CO₂** — estimated vs actual emissions (converted to grams)
        - **Energy** — estimated energy (TDP × time) vs actual energy_consumed
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
        runtime_error_pct = (
            abs(est_runtime - act_runtime) / act_runtime * 100.0
            if act_runtime
            else None
        )
        co2_error_pct = (
            abs(est_co2_g - act_co2_g) / act_co2_g * 100.0
            if act_co2_g
            else None
        )
        runtime_ratio = (
            est_runtime / act_runtime if act_runtime else None
        )
        co2_ratio = est_co2_g / act_co2_g if act_co2_g else None

        energy_error_pct = None
        energy_ratio = None
        if est_energy_kwh is not None and act_energy_kwh:
            energy_error_pct = (
                abs(est_energy_kwh - act_energy_kwh) / act_energy_kwh * 100.0
            )
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

    def _load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        path = Path(self.output_dir) / filename
        if not path.exists():
            logger.info("File not found: %s", path)
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    def _save_json(self, comparison: Dict[str, Any]) -> None:
        path = Path(self.output_dir) / EMISSION_COMPARISON_FILENAME
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(comparison, f, indent=4)
        logger.info("Emission comparison saved to %s", path)

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

    def _log_comparison_to_wandb(self, comparison: Dict[str, Any]) -> None:
        try:

            run = wandb.init(
                project=self.wandb_project,
                name=f"{self.wandb_name}-comparison",
                job_type="emission-comparison",
                tags=["emission-comparison"],
                reinit=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise wandb run for comparison: %s", exc
            )
            return

        try:
            wandb_keys = {
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
            for src_key, wandb_key in wandb_keys.items():
                value = comparison.get(src_key)
                if value is not None:
                    run.summary[wandb_key] = value

            logger.info(
                "Emission comparison logged to wandb run '%s'.", run.name
            )
        finally:
            run.finish()
