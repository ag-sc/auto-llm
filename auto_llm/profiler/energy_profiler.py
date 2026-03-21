import dataclasses
import logging
import os
from codecarbon import EmissionsTracker
import wandb

from auto_llm.constants import EMISSIONS_CSV_FILENAME


logger = logging.getLogger(__name__)


class EnergyProfiler:
    """Context manager wrapping CodeCarbon's ``EmissionsTracker`` to profile
    energy consumption and CO₂ emissions during training or evaluation.

    Args:
        output_dir: Directory where the ``emissions.csv`` file is written.
        project_name: Label stored in the CSV ``project_name`` column.
            Passed to ``EmissionsTracker.project_name``. Also used as the
            wandb project when ``log_to_wandb`` is ``True``.
        experiment_name: Human-readable experiment label used in log
            messages and as the base for the dedicated wandb run name
            (suffixed with ``-energy``).
        is_main_process: When ``False`` the profiler is a no-op. Set this to
            ``accelerator.is_main_process`` in multi-GPU setups so that only
            rank-0 tracks emissions.
        measure_power_secs: Sampling interval in seconds for hardware power
            readings. Passed to ``EmissionsTracker.measure_power_secs``.
            Lower values give finer granularity at the cost of higher
            overhead.
        log_to_wandb: When ``True`` a **dedicated** wandb run is created in
            ``__exit__`` to log emission metrics. The run is named
            ``{experiment_name}-energy`` and placed in ``project_name``.
            Defaults to ``False`` (no wandb logging).

    Usage::

        # Training
        with EnergyProfiler(
            output_dir="/out/model",
            project_name="my-project",
            experiment_name="sft-run-1",
            is_main_process=accelerator.is_main_process,
            log_to_wandb=True,
        ):
            trainer.train()

        # Evaluation
        with EnergyProfiler(
            output_dir="/out/eval",
            project_name="llm4kmu-eval",
            experiment_name="pico-run",
            log_to_wandb=True,
        ):
            cli_evaluate(args=lm_eval_args)
    """

    def __init__(
        self,
        output_dir: str,
        project_name: str = "auto-llm",
        experiment_name: str = "run",
        is_main_process: bool = True,
        measure_power_secs: int = 15,
        log_to_wandb: bool = False,
    ):
        self.output_dir = output_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.is_main_process = is_main_process
        self.measure_power_secs = measure_power_secs
        self.log_to_wandb = log_to_wandb
        self._tracker = None
        self._final_emissions: dict = None

    @property
    def final_emissions_data(self) -> dict:
        """Actual emissions as a plain dict (available after ``__exit__``).

        Populated from ``EmissionsTracker.final_emissions_data`` via
        ``dataclasses.asdict`` so that all values are native Python types.
        Returns ``None`` if the profiler hasn't run or was a no-op.
        """
        return self._final_emissions

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
            allow_multiple_runs=True,
            log_level="warning",
        )
        self._tracker.start()
        logger.info(
            "Energy profiling started (project=%s, experiment=%s)",
            self.project_name,
            self.experiment_name,
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self._tracker is None:
            return False

        emissions_total = self._tracker.stop()
        logger.info(
            "Energy profiling stopped — total emissions: %.6f kg CO₂eq",
            emissions_total,
        )

        # Capture final data as a plain dict for downstream consumers
        raw = getattr(self._tracker, "final_emissions_data", None)
        if raw is not None:
            self._final_emissions = dataclasses.asdict(raw)

        self._log_to_wandb()
        return False  # do not suppress exceptions

    def _log_to_wandb(self):
        """Create a dedicated wandb run and log emissions metrics.

        Reads metrics directly from ``EmissionsTracker.final_emissions_data``
        (an ``EmissionsData`` dataclass populated by ``tracker.stop()``) rather
        than re-parsing the CSV file. This is faster, avoids file-I/O race
        conditions, and preserves native Python types (no string casting).
        """

        if not self.log_to_wandb:
            logger.info("wandb logging disabled — skipping emissions logging.")
            return

        emissions_data = self._final_emissions
        if emissions_data is None:
            logger.warning(
                "EmissionsTracker did not produce final_emissions_data — "
                "skipping wandb logging."
            )
            return

        try:
            run = wandb.init(
                project=self.project_name,
                name=f"{self.experiment_name}-energy",
                job_type="energy-profiling",
                tags=["energy-profiling"],
                reinit=True,
            )
        except Exception as e:
            logger.warning("Failed to initialise dedicated wandb run: %s", e)
            return

        # Map EmissionsData fields → wandb summary keys.
        # Uses the in-memory dataclass directly (no CSV parsing needed).
        key_map = {
            # Energy & emissions
            "energy_consumed": "emissions/energy_consumed_kWh",
            "emissions": "emissions/emissions_kg",
            "emissions_rate": "emissions/emissions_rate_kg_per_s",
            "cpu_energy": "emissions/cpu_energy_kWh",
            "gpu_energy": "emissions/gpu_energy_kWh",
            "ram_energy": "emissions/ram_energy_kWh",
            "water_consumed": "emissions/water_consumed_L",
            # Power draw (mean)
            "cpu_power": "emissions/cpu_power_W",
            "gpu_power": "emissions/gpu_power_W",
            "ram_power": "emissions/ram_power_W",
            # Duration
            "duration": "emissions/duration_s",
            # Utilization
            "cpu_utilization_percent": "emissions/cpu_utilization_percent",
            "gpu_utilization_percent": "emissions/gpu_utilization_percent",
            "ram_utilization_percent": "emissions/ram_utilization_percent",
            "ram_used_gb": "emissions/ram_used_gb",
            # Location
            "country_name": "emissions/country_name",
            "country_iso_code": "emissions/country_iso_code",
            "region": "emissions/region",
            "cloud_provider": "emissions/cloud_provider",
            "cloud_region": "emissions/cloud_region",
            "on_cloud": "emissions/on_cloud",
            # Hardware
            "cpu_count": "emissions/cpu_count",
            "cpu_model": "emissions/cpu_model",
            "gpu_count": "emissions/gpu_count",
            "gpu_model": "emissions/gpu_model",
            "ram_total_size": "emissions/ram_total_size_GB",
            # Tracking config
            "tracking_mode": "emissions/tracking_mode",
            "pue": "emissions/pue",
            # System info
            "os": "emissions/os",
            "python_version": "emissions/python_version",
            "codecarbon_version": "emissions/codecarbon_version",
        }

        try:
            for field, wandb_key in key_map.items():
                value = emissions_data.get(field)
                if value is None or value == "":
                    continue
                run.summary[wandb_key] = value

            logger.info("Emissions metrics logged to dedicated wandb run '%s'.", run.name)
        finally:
            run.finish()



