import dataclasses
import logging
import os
from typing import Any, Dict, Optional
from codecarbon import EmissionsTracker

from auto_llm.constants import EMISSIONS_CSV_FILENAME


logger = logging.getLogger(__name__)


# Threshold above which cpu_power is considered suspicious and a
# warning is emitted.  Most single-socket server CPUs have a TDP
# well below 500 W; values above this usually indicate CodeCarbon's
# CPU-load fallback is reading whole-machine load instead of per-
# process load (e.g. RAPL files not accessible).
_CPU_POWER_SANITY_THRESHOLD_W = 500


class EnergyProfiler:
    """Context manager wrapping CodeCarbon's ``EmissionsTracker`` to profile
    energy consumption and CO₂ emissions during training or evaluation.

    After the context manager exits, call :meth:`get_wandb_metrics` to
    retrieve a ``{wandb_key: value}`` dict suitable for logging via
    :class:`~auto_llm.profiler.wandb_energy_logger.WandbEnergyLogger`.

    Args:
        output_dir: Directory where the ``emissions.csv`` file is written.
        project_name: Label stored in the CSV ``project_name`` column.
            Passed to ``EmissionsTracker.project_name``.
        experiment_name: Human-readable experiment label used in log
            messages.
        is_main_process: When ``False`` the profiler is a no-op. Set this to
            ``accelerator.is_main_process`` in multi-GPU setups so that only
            rank-0 tracks emissions.
        measure_power_secs: Sampling interval in seconds for hardware power
            readings. Passed to ``EmissionsTracker.measure_power_secs``.
            Lower values give finer granularity at the cost of higher
            overhead.
        tracking_mode: One of ``"machine"`` or ``"process"``.

            * ``"machine"`` (default) — reads whole-node CPU load via
              ``psutil.cpu_percent()`` and scales by TDP.  The load
              factor is bounded (0–100 %), so reported power never
              exceeds the configured TDP.  **Strongly recommended**
              when combined with ``force_cpu_power`` on clusters where
              ``lscpu`` reports a virtualised socket count.
            * ``"process"`` — tracks only the current process tree via
              ``psutil.Process.cpu_times()``.  Can produce unbounded
              power values when child processes accumulate more CPU
              time than wall-clock time (common with data-loader
              workers on cgroup-restricted SLURM jobs).  Use only on
              bare-metal single-user machines with accurate
              ``cpu_count``.

            .. note::
               ``tracking_mode`` only affects **CPU power** measurement
               when RAPL is unavailable and CodeCarbon falls back to
               "CPU-load" mode.  **GPU** tracking always uses NVML
               regardless of this setting (scope GPUs via
               ``CUDA_VISIBLE_DEVICES`` or CodeCarbon's ``gpu_ids``).
               **RAM** uses a DIMM-count heuristic in both modes —
               override with ``force_ram_power`` if needed.

        force_cpu_power: Override CPU TDP (watts) used by CodeCarbon's
            fallback CPU-load estimator.  Useful when RAPL is not
            available and the auto-detected TDP is wrong.  ``None``
            means auto-detect.
        force_ram_power: Override RAM power (watts).  Estimate with
            ``sudo lshw -C memory -short | grep DIMM`` then multiply
            slots × 5 W.  ``None`` means use CodeCarbon's heuristic.

    Usage::

        with EnergyProfiler(
            output_dir="/out/model",
            project_name="my-project",
            experiment_name="sft-run-1",
            is_main_process=accelerator.is_main_process,
        ) as profiler:
            trainer.train()

        metrics = profiler.get_wandb_metrics()  # pass to WandbEnergyLogger
    """

    def __init__(
        self,
        output_dir: str,
        project_name: str = "auto-llm",
        experiment_name: str = "run",
        is_main_process: bool = True,
        measure_power_secs: int = 15,
        tracking_mode: str = "machine",
        force_cpu_power: Optional[int] = None,
        force_ram_power: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.is_main_process = is_main_process
        self.measure_power_secs = measure_power_secs
        self.tracking_mode = tracking_mode
        self.force_cpu_power = force_cpu_power
        self.force_ram_power = force_ram_power
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
        #   tracking_mode — "process" scopes CPU measurement to this
        #       process tree (recommended on shared clusters where RAPL
        #       is unavailable); "machine" reads whole-node power
        #       (use only when the job has exclusive node access).
        #   force_cpu_power / force_ram_power — optional user overrides
        #       for CPU TDP and RAM power when auto-detection is wrong.
        #   save_to_file=True  — persist results to the CSV.
        #   save_to_api=False  — do not push to the CodeCarbon dashboard.
        #   log_level="warning" — suppress CodeCarbon's verbose INFO logs.
        tracker_kwargs = dict(
            project_name=self.project_name,
            output_dir=self.output_dir,
            output_file=EMISSIONS_CSV_FILENAME,
            tracking_mode=self.tracking_mode,
            measure_power_secs=self.measure_power_secs,
            save_to_file=True,
            save_to_api=False,
            allow_multiple_runs=True,
            log_level="warning",
        )
        if self.force_cpu_power is not None:
            tracker_kwargs["force_cpu_power"] = self.force_cpu_power
        if self.force_ram_power is not None:
            tracker_kwargs["force_ram_power"] = self.force_ram_power

        self._tracker = EmissionsTracker(**tracker_kwargs)
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

        # Sanity-check: warn about suspiciously high CPU power which
        # typically indicates that RAPL was unavailable and CodeCarbon's
        # CPU-load fallback is reading whole-machine load.
        if self._final_emissions is not None:
            cpu_power = self._final_emissions.get("cpu_power")
            if (
                cpu_power is not None
                and cpu_power > _CPU_POWER_SANITY_THRESHOLD_W
            ):
                logger.warning(
                    "Measured CPU power (%.1f W) exceeds %d W — this "
                    "usually means RAPL is unavailable and CodeCarbon "
                    "fell back to CPU-load mode on the whole machine. "
                    "Consider setting tracking_mode='process' or "
                    "providing force_cpu_power to cap the TDP.",
                    cpu_power,
                    _CPU_POWER_SANITY_THRESHOLD_W,
                )

        return False  # do not suppress exceptions

    def get_wandb_metrics(self) -> Optional[Dict[str, Any]]:
        """Return emissions data as a wandb-ready ``{wandb_key: value}`` dict.

        Returns ``None`` if no emissions data is available (profiler hasn't
        run or was a no-op).  This method does **not** create a wandb run —
        pass the result to
        :meth:`~auto_llm.profiler.wandb_energy_logger.WandbEnergyLogger.log`.
        """
        emissions_data = self._final_emissions
        if emissions_data is None:
            return None

        summary: Dict[str, Any] = {}
        for field, wandb_key in self._EMISSIONS_KEY_MAP.items():
            value = emissions_data.get(field)
            if value is not None and value != "":
                summary[wandb_key] = value
        return summary

    # Map EmissionsData fields → wandb summary keys.
    _EMISSIONS_KEY_MAP = {
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
