"""Orchestrates the full FLOPs → Runtime → Emission estimator chain.

``EstimationPipeline`` replaces the old procedural ``run_and_save_estimate()``
function.  It builds the three-stage estimator chain, runs all estimates, and
optionally persists the result as ``emission_estimate.json``.

Usage::

    # From the trainer (passing a TrainerRunConfig object):
    pipeline = EstimationPipeline(
        output_dir="/out/model",
        gpu_name="NVIDIA A40",
        is_eval=False,
        config=trainer_run_config,
    )
    result = pipeline.run()   # estimate + save (best-effort)

    # From the evaluator (passing a YAML path):
    pipeline = EstimationPipeline(
        output_dir="/out/eval",
        gpu_name=None,          # auto-detect
        is_eval=True,
        config_path="config.yaml",
    )
    result = pipeline.run()

    # From the UI (estimate only, no persistence):
    pipeline = EstimationPipeline(
        output_dir=None,
        gpu_name="NVIDIA A40",
        is_eval=False,
        config_path="config.yaml",
    )
    estimate = pipeline.estimate()
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

from auto_llm.constants import (
    DEFAULT_CARBON_INTENSITY_G_PER_KWH,
    EMISSION_ESTIMATE_FILENAME,
)
from auto_llm.estimator.emission_estimator import EmissionEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.utils import get_gpu_params, get_model_params, resolve_gpu_name

if TYPE_CHECKING:
    from auto_llm.dto.trainer_run_config import TrainerRunConfig

logger = logging.getLogger(__name__)


class EstimationPipeline:
    """Build and execute the FLOPs → Runtime → Emission estimator chain.

    Accepts either a ``config_path`` (YAML file path) **or** a ``config``
    (``TrainerRunConfig`` object) for training estimation.  For evaluation
    estimation ``config_path`` is required.

    Args:
        output_dir: Directory where ``emission_estimate.json`` will be
            written by :meth:`save`.  Can be ``None`` when only
            :meth:`estimate` is needed (e.g. the UI).
        gpu_name: Explicit GPU name.  ``None`` triggers auto-detection.
        is_eval: ``True`` for inference estimation, ``False`` for training.
        config_path: Path to the YAML config file (evaluator or trainer).
        config: A ``TrainerRunConfig`` instance — used by the trainer
            wrapper to avoid re-reading the YAML from disk.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        gpu_name: Optional[str] = None,
        is_eval: bool = False,
        config_path: Optional[str] = None,
        config: Optional[TrainerRunConfig] = None,
    ):
        self.output_dir = output_dir
        self.gpu_name = gpu_name
        self.is_eval = is_eval
        self.config_path = config_path
        self.config = config

    def estimate(self) -> Dict[str, Any]:
        """Build the estimator chain, execute it, and return the result dict.

        Raises:
            ValueError: When the required configuration is missing.
            Exception: When GPU resolution or estimator construction fails.

        Returns:
            A dict containing ``estimated_flops``, ``estimated_runtime_s``,
            ``estimated_co2_g``, ``gpu_name``, ``carbon_intensity_g_per_kWh``,
            and ``timestamp``.
        """
        gpu_params = get_gpu_params()
        resolved_gpu = resolve_gpu_name(
            gpu_name=self.gpu_name, gpu_params=gpu_params
        )
        if resolved_gpu is None:
            raise ValueError(
                "Could not resolve GPU name — energy estimation aborted."
            )

        models_meta = get_model_params()
        flops_estimator = self._build_flops_estimator(models_meta)

        runtime_estimator = RuntimeEstimator(
            flops_estimator=flops_estimator,
            gpu_params=gpu_params,
            gpu_name=resolved_gpu,
        )
        emission_estimator = EmissionEstimator(
            runtime_estimator=runtime_estimator,
            gpu_params=gpu_params,
            gpu_name=resolved_gpu,
        )

        estimated_flops = flops_estimator.estimate()
        estimated_runtime_s = runtime_estimator.estimate()
        estimated_co2_g = emission_estimator.estimate()
        estimated_energy_kwh = emission_estimator.estimate_energy_kwh()

        return {
            "estimated_flops": estimated_flops,
            "estimated_runtime_s": estimated_runtime_s,
            "estimated_co2_g": estimated_co2_g,
            "estimated_energy_kwh": estimated_energy_kwh,
            "gpu_name": resolved_gpu,
            "carbon_intensity_g_per_kWh": DEFAULT_CARBON_INTENSITY_G_PER_KWH,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def save(self, estimate: Dict[str, Any]) -> None:
        """Persist *estimate* as ``emission_estimate.json`` in *output_dir*.

        Raises:
            ValueError: When ``output_dir`` was not set.
        """
        if not self.output_dir:
            raise ValueError("output_dir is required for saving estimates.")

        os.makedirs(self.output_dir, exist_ok=True)
        estimate_path = os.path.join(self.output_dir, EMISSION_ESTIMATE_FILENAME)
        with open(estimate_path, "w") as f:
            json.dump(estimate, f, indent=4)
        logger.info(
            "Pre-run energy estimate saved to %s "
            "(FLOPs=%s, runtime=%.2fs, CO₂=%.2fg)",
            estimate_path,
            estimate["estimated_flops"],
            estimate["estimated_runtime_s"],
            estimate["estimated_co2_g"],
        )

    def run(self) -> Optional[Dict[str, Any]]:
        """Execute :meth:`estimate` then :meth:`save` — best-effort.

        Returns:
            The estimate dict on success, or ``None`` on failure.
        """
        try:
            estimate = self.estimate()
            self.save(estimate)
            return estimate
        except Exception as exc:
            logger.warning("Energy estimation failed (best-effort) — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_flops_estimator(self, models_meta: Dict[str, Any]):
        """Construct the appropriate FLOPs estimator."""
        if self.is_eval:
            from auto_llm.estimator.inference_flops_estimator import (
                InferenceFlopsEstimator,
            )

            if self.config_path is None:
                raise ValueError(
                    "config_path is required for evaluation estimation."
                )
            return InferenceFlopsEstimator(
                config_path=self.config_path, models_meta=models_meta
            )

        # Training
        from auto_llm.estimator.trainer_flops_estimator import (
            TrainerFlopsEstimator,
        )

        if self.config is not None:
            return TrainerFlopsEstimator(
                config=self.config, models_meta=models_meta
            )
        if self.config_path is not None:
            return TrainerFlopsEstimator(
                config_path=self.config_path, models_meta=models_meta
            )

        raise ValueError(
            "Either config or config_path is required for training estimation."
        )
