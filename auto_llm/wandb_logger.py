import logging
from typing import Any, Dict, List, Optional

import wandb

_logger = logging.getLogger(__name__)


class WandbLogger:
    """
        Base wandb logger — accumulate metrics and flush to a single run.
        Accumulate ``{key: value}`` metrics and log them in one wandb run.

    Subclass this to create domain-specific loggers (e.g.
    :class:`~auto_llm.profiler.wandb_energy_logger.WandbEnergyLogger`).


        Args:
            project: Wandb project name.
            name: Base run name.  If *name_suffix* is set the run is created as
                ``{name}{name_suffix}``.
            tags: Wandb tags attached to the run.
            job_type: Wandb ``job_type`` for the run (default ``"logging"``).
            name_suffix: Optional suffix appended to *name* when creating the
                wandb run (e.g. ``"-energy"``).

        Usage::

        logger = WandbLogger(project="my-project", name="run-1")
        logger.log({"metric/loss": 0.42})
        logger.flush()
    """

    def __init__(
        self,
        project: str,
        name: str,
        tags: Optional[List[str]] = None,
        job_type: str = "logging",
        name_suffix: str = "",
    ):
        self.project = project
        self.name = name
        self.tags = list(tags) if tags else []
        self.job_type = job_type
        self.name_suffix = name_suffix
        self._metrics: Dict[str, Any] = {}

    @property
    def run_name(self) -> str:
        """Full run name including the suffix."""
        return f"{self.name}{self.name_suffix}"

    def log(self, metrics: Optional[Dict[str, Any]]) -> None:
        """Merge *metrics* into the internal accumulator.

        ``None`` values and ``None`` dicts are silently skipped so callers
        don't need to guard against missing data.
        """
        if not metrics:
            return
        for key, value in metrics.items():
            if value is not None:
                self._metrics[key] = value

    def flush(self) -> None:
        """Create a wandb run, write all accumulated metrics, and finish.

        No-ops (with a debug log) when no metrics have been collected.
        """
        if not self._metrics:
            _logger.debug(
                "%s: no metrics collected — skipping flush.",
                self.__class__.__name__,
            )
            return

        try:
            run = wandb.init(
                project=self.project,
                name=self.run_name,
                job_type=self.job_type,
                tags=self.tags,
                reinit=True,
            )
        except Exception as exc:
            _logger.warning(
                "Failed to initialise wandb run '%s': %s",
                self.run_name,
                exc,
            )
            return

        try:
            run.summary.update(self._metrics)
            _logger.info(
                "Metrics logged to wandb run '%s' (%d keys).",
                run.name,
                len(self._metrics),
            )
        finally:
            run.finish()
