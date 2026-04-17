"""Project-level Pareto frontier plot on wandb.

Backfills ``pareto/*`` fields into every energy run's summary so a workspace
ScatterPlot panel can aggregate them across the whole project. A helper also
upserts the panel itself via the ``wandb-workspaces`` SDK.

Typical usage (see also ``scripts/wandb_pareto_plot.py``)::

    from auto_llm.evaluator.plots.wandb_pareto_plot import (
        backfill_pareto_flags,
        ensure_project_scatter_panel,
    )

    backfill_pareto_flags(entity="llm4kmu", project="open-medical-llm-energy")
    ensure_project_scatter_panel(entity="llm4kmu", project="open-medical-llm-energy")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import wandb

from auto_llm.evaluator.plots.pareto_plot import compute_pareto_indices

_logger = logging.getLogger(__name__)


DEFAULT_ENERGY_KEY = "emissions/energy_consumed_kWh"
DEFAULT_SCORE_KEY = "eval/avg_score"
DEFAULT_TAG = "energy-profiling"
DEFAULT_PANEL_TITLE = "Energy vs Accuracy — Pareto Frontier"
DEFAULT_SECTION_NAME = "Pareto Frontier"

PARETO_ENERGY_WH_KEY = "pareto/energy_wh"
PARETO_ACCURACY_KEY = "pareto/accuracy_pct"
PARETO_IS_OPTIMAL_KEY = "pareto/is_optimal"
PARETO_RANK_KEY = "pareto/rank"


@dataclass
class ParetoPoint:
    name: str
    run_id: str
    energy_wh: float
    accuracy_pct: float
    is_pareto: bool = False
    rank: int = -1


def _get_summary_value(summary: Any, key: str) -> Optional[float]:
    """Read a scalar out of a wandb summary (supports nested ``a/b`` keys)."""
    if key in summary:
        value = summary[key]
    else:
        node: Any = summary
        for part in key.split("/"):
            if node is None:
                return None
            try:
                node = node[part]
            except (KeyError, TypeError):
                return None
        value = node

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def extract_pareto_points(
    runs: Iterable[Any],
    energy_key: str = DEFAULT_ENERGY_KEY,
    score_key: str = DEFAULT_SCORE_KEY,
    score_scale: float = 100.0,
) -> List[ParetoPoint]:
    """Pull ``(energy_wh, accuracy_pct)`` from each run and mark Pareto points."""
    points: List[ParetoPoint] = []
    for run in runs:
        summary = run.summary
        energy_kwh = _get_summary_value(summary, energy_key)
        score = _get_summary_value(summary, score_key)
        if energy_kwh is None or score is None:
            _logger.debug(
                "Skipping run %s: missing %s or %s", run.name, energy_key, score_key
            )
            continue
        points.append(
            ParetoPoint(
                name=run.name,
                run_id=run.id,
                energy_wh=energy_kwh * 1000.0,
                accuracy_pct=score * score_scale,
            )
        )

    if not points:
        return []

    energies = np.array([p.energy_wh for p in points], dtype=float)
    accuracies = np.array([p.accuracy_pct for p in points], dtype=float)
    frontier_indices = compute_pareto_indices(energies, accuracies)

    frontier_order = sorted(frontier_indices, key=lambda i: energies[i])
    for rank, idx in enumerate(frontier_order):
        points[idx].is_pareto = True
        points[idx].rank = rank

    return points


def _filter_runs_by_tag(runs: Iterable[Any], tag: Optional[str]) -> List[Any]:
    if not tag:
        return list(runs)
    return [r for r in runs if tag in (r.tags or [])]


def backfill_pareto_flags(
    entity: str,
    project: str,
    energy_key: str = DEFAULT_ENERGY_KEY,
    score_key: str = DEFAULT_SCORE_KEY,
    score_scale: float = 100.0,
    tag: Optional[str] = DEFAULT_TAG,
    dry_run: bool = False,
    api: Optional[wandb.Api] = None,
) -> List[ParetoPoint]:
    """Write ``pareto/*`` fields into each qualifying run's summary.

    Returns the list of points (with is_pareto / rank populated). When
    ``dry_run`` is set, nothing is persisted — useful for previewing the
    frontier before writing.
    """
    api = api or wandb.Api()
    all_runs = list(api.runs(f"{entity}/{project}"))
    runs = _filter_runs_by_tag(all_runs, tag)
    if not runs:
        _logger.warning(
            "No runs found in %s/%s with tag %r", entity, project, tag
        )
        return []

    points = extract_pareto_points(
        runs, energy_key=energy_key, score_key=score_key, score_scale=score_scale
    )
    if not points:
        _logger.warning(
            "No runs expose both %s and %s", energy_key, score_key
        )
        return []

    runs_by_id = {r.id: r for r in runs}
    written = 0
    for point in points:
        run = runs_by_id[point.run_id]
        update: Dict[str, Any] = {
            PARETO_ENERGY_WH_KEY: point.energy_wh,
            PARETO_ACCURACY_KEY: point.accuracy_pct,
            PARETO_IS_OPTIMAL_KEY: bool(point.is_pareto),
            PARETO_RANK_KEY: int(point.rank),
        }
        if dry_run:
            flag = "pareto" if point.is_pareto else "       "
            print(
                f"  [{flag}] {run.name}: energy={point.energy_wh:.2f} Wh, "
                f"acc={point.accuracy_pct:.2f}%, rank={point.rank}"
            )
            continue
        run.summary.update(update)
        run.update()
        written += 1

    if not dry_run:
        _logger.info(
            "Backfilled pareto/* on %d runs in %s/%s", written, entity, project
        )
    return points


def ensure_project_scatter_panel(
    entity: str,
    project: str,
    panel_title: str = DEFAULT_PANEL_TITLE,
    section_name: str = DEFAULT_SECTION_NAME,
    workspace_name: str = "Pareto Frontier",
) -> str:
    """Upsert a project-level scatter panel that plots pareto/* summary fields.

    Creates (or replaces, if it already exists by name) a saved workspace
    view containing one ScatterPlot panel. Returns the workspace URL.
    """
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    scatter = wr.ScatterPlot(
        title=panel_title,
        x=wr.SummaryMetric(name=PARETO_ENERGY_WH_KEY),
        y=wr.SummaryMetric(name=PARETO_ACCURACY_KEY),
        regression=False,
    )

    workspace = ws.Workspace(
        name=workspace_name,
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name=section_name,
                panels=[scatter],
                is_open=True,
            ),
        ],
    )

    saved = workspace.save()
    url = getattr(saved, "url", None) or getattr(workspace, "url", "")
    _logger.info("Workspace upserted: %s", url)
    return url
