from typing import List, Optional

from auto_llm.wandb_logger import WandbLogger


class WandbEnergyLogger(WandbLogger):
    """Wandb logger for energy-profiling metrics.

    Collects metrics from multiple sources (``EstimationPipeline``,
    ``EnergyProfiler``, ``EmissionComparator``) and flushes them into a
    **single** dedicated wandb run.

        Defaults:

        - ``job_type`` → ``"energy-profiling"``
        - Run name is ``{name}-energy``
        - ``"energy-profiling"`` tag is always present

        Args:
            project: Wandb project name.
            name: Base run name — the run is created as ``{name}-energy``.
            tags: Additional wandb tags (``"energy-profiling"`` is always
                included).

        Usage::

        logger = WandbEnergyLogger(project="my-project", name="sft-run-1")
        logger.log(pipeline.get_wandb_metrics())
        logger.log(profiler.get_wandb_metrics())
        logger.log(comparator.get_wandb_metrics())
        logger.flush()
    """

    def __init__(
        self,
        project: str,
        name: str,
        tags: Optional[List[str]] = None,
    ):
        resolved_tags = list(tags) if tags else []
        if "energy-profiling" not in resolved_tags:
            resolved_tags.insert(0, "energy-profiling")

        super().__init__(
            project=project,
            name=name,
            tags=resolved_tags,
            job_type="energy-profiling",
            name_suffix="-energy",
        )
