import csv
from accelerate.logging import get_logger
import os
from pathlib import Path
from codecarbon import EmissionsTracker
import wandb



logger = get_logger(__name__)

EMISSIONS_CSV_FILENAME = "emissions.csv"


class EnergyProfiler:
    """Context manager wrapping CodeCarbon's ``EmissionsTracker`` to profile
    energy consumption and CO₂ emissions during training or evaluation.

    Args:
        output_dir: Directory where the ``emissions.csv`` file is written.
        project_name: Label stored in the CSV ``project_name`` column.
            Passed to ``EmissionsTracker.project_name``.
        experiment_name: Human-readable experiment label used only in log
            messages (not written to the CSV by CodeCarbon).
        is_main_process: When ``False`` the profiler is a no-op. Set this to
            ``accelerator.is_main_process`` in multi-GPU setups so that only
            rank-0 tracks emissions.
        measure_power_secs: Sampling interval in seconds for hardware power
            readings. Passed to ``EmissionsTracker.measure_power_secs``.
            Lower values give finer granularity at the cost of higher
            overhead.

    Usage::

        # Training
        with EnergyProfiler(
            output_dir="/out/model",
            project_name="my-project",
            experiment_name="sft-run-1",
            is_main_process=accelerator.is_main_process,
        ):
            trainer.train()

        # Evaluation
        with EnergyProfiler(output_dir="/out/eval"):
            cli_evaluate(args=lm_eval_args)
    """

    def __init__(
        self,
        output_dir: str,
        project_name: str = "auto-llm",
        experiment_name: str = "run",
        is_main_process: bool = True,
        measure_power_secs: int = 15,
    ):
        self.output_dir = output_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.is_main_process = is_main_process
        self.measure_power_secs = measure_power_secs
        self._tracker = None

    def __enter__(self):
        if not self.is_main_process:
            return self

        os.makedirs(self.output_dir, exist_ok=True)

        # EmissionsTracker configuration:
        #   tracking_mode="machine" — reads total power draw from all GPUs
        #       on the node via the driver (nvidia-smi / NVML), rather than
        #       isolating per-process consumption. Preferred for full-node
        #       training/eval jobs.
        #   save_to_file=True  — persist results to the CSV.
        #   save_to_api=False  — do not push to the CodeCarbon dashboard.
        #   log_level="warning" — suppress CodeCarbon's verbose INFO logs.
        self._tracker = EmissionsTracker(
            project_name=self.project_name,
            output_dir=self.output_dir,
            output_file=EMISSIONS_CSV_FILENAME,
            tracking_mode="machine",
            measure_power_secs=self.measure_power_secs,
            save_to_file=True,
            save_to_api=False,
            log_level="warning",
        )
        self._tracker.start()
        logger.info(
            "Energy profiling started (project=%s, experiment=%s)",
            self.project_name,
            self.experiment_name,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tracker is None:
            return False

        emissions_total = self._tracker.stop()
        logger.info(
            "Energy profiling stopped — total emissions: %.6f kg CO₂eq",
            emissions_total,
        )
        self._log_to_wandb()
        return False  # do not suppress exceptions

   
    def _log_to_wandb(self):
        """Read the last row of the emissions CSV and push it to wandb.run.summary."""

        if wandb.run is None:
            logger.warning("No active wandb run — skipping emissions logging to wandb.")
            return

        emissions_data = self._read_last_emissions_row()
        if emissions_data is None:
            logger.warning("Could not read emissions CSV — skipping wandb logging.")
            return

        # Map CSV columns → wandb summary keys
        key_map = {
            "energy_consumed": "emissions/energy_consumed_kWh",
            "emissions": "emissions/emissions_kg",
            "emissions_rate": "emissions/emissions_rate_kg_per_s",
            "cpu_power": "emissions/cpu_power_W",
            "gpu_power": "emissions/gpu_power_W",
            "ram_power": "emissions/ram_power_W",
            "duration": "emissions/duration_s",
            "country_iso_code": "emissions/country_iso_code",
            "region": "emissions/region",
            "cloud_provider": "emissions/cloud_provider",
            "cloud_region": "emissions/cloud_region",
            "cpu_count": "emissions/cpu_count",
            "gpu_count": "emissions/gpu_count",
            "gpu_model": "emissions/gpu_model",
        }

        for csv_col, wandb_key in key_map.items():
            value = emissions_data.get(csv_col)
            if value is None or value == "":
                continue
            # Try to cast numeric values
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            wandb.run.summary[wandb_key] = value

        logger.info("Emissions metrics logged to wandb run '%s'.", wandb.run.name)


    def _read_last_emissions_row(self) -> dict | None:
        """Return the last row of the emissions CSV as a dict, or None on failure."""
        csv_path = Path(self.output_dir) / EMISSIONS_CSV_FILENAME
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return None
            return rows[-1]
        except Exception as e:
            logger.warning("Failed to parse emissions CSV: %s", e)
            return None
