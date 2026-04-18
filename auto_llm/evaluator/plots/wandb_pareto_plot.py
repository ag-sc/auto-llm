"""Project-level Pareto frontier plot on wandb.

Backfills ``pareto/*`` fields into every energy run's summary so a workspace
custom chart panel can aggregate them across the whole project.  The chart
uses a layered Vega-Lite spec (scatter + black dashed Pareto frontier line)
registered as a reusable chart preset via ``wandb.Api().create_custom_chart``.

Typical usage (see also ``scripts/wandb_pareto_plot.py``)::

    from auto_llm.evaluator.plots.wandb_pareto_plot import (
        backfill_pareto_flags,
        ensure_chart_preset,
        ensure_project_scatter_panel,
    )

    backfill_pareto_flags(entity="llm4kmu", project="open-medical-llm-energy")
    ensure_chart_preset(entity="llm4kmu")
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
PARETO_NAME_KEY = "pareto/name"

DEFAULT_CHART_PRESET_NAME = "pareto-frontier"

# ---------------------------------------------------------------------------
# Vega-Lite specification — layered scatter + Pareto frontier dashed line
# ---------------------------------------------------------------------------
# Field placeholders (``${field:...}``) are resolved by wandb at render time
# from the ``chart_fields`` mapping supplied when the panel is created.
#
# Layer 1 — line:  black dashed line connecting only Pareto-optimal points,
#           ordered left-to-right by energy (ascending).
# Layer 2 — ideal star: a single ★ marker at (min energy, max accuracy)
#           computed via Vega-Lite aggregate transforms.
# Layer 3 — scatter: all runs as filled circles with tooltips.
#           Rendered last so it sits on top and receives hover events.
# ---------------------------------------------------------------------------
PARETO_VEGA_SPEC: dict = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"name": "wandb"},
    "title": "${string:title}",
    "layer": [
        # --- Layer 1: Pareto frontier dashed line ---
        {
            "transform": [
                {"filter": {"field": "${field:is_pareto}", "equal": True}}
            ],
            "mark": {
                "type": "line",
                "color": "black",
                "strokeWidth": 2,
                "strokeDash": [8, 4],
                "point": False,
            },
            "encoding": {
                "x": {
                    "field": "${field:energy}",
                    "type": "quantitative",
                },
                "y": {
                    "field": "${field:accuracy}",
                    "type": "quantitative",
                },
                "order": {
                    "field": "${field:energy}",
                    "type": "quantitative",
                },
            },
        },
        # --- Layer 2: ideal point (★) ---
        {
            "transform": [
                {
                    "aggregate": [
                        {"op": "min", "field": "${field:energy}", "as": "min_energy"},
                        {"op": "max", "field": "${field:accuracy}", "as": "max_accuracy"},
                    ]
                }
            ],
            "mark": {
                "type": "point",
                "shape": "cross",
                "size": 200,
                "color": "black",
                "filled": True,
                "strokeWidth": 2,
            },
            "encoding": {
                "x": {"field": "min_energy", "type": "quantitative"},
                "y": {"field": "max_accuracy", "type": "quantitative"},
            },
        },
        # --- Layer 3: scatter (all runs) — on top for hover ---
        {
            "mark": {
                "type": "point",
                "filled": True,
                "size": 100,
                "opacity": 0.85,
            },
            "encoding": {
                "x": {
                    "field": "${field:energy}",
                    "type": "quantitative",
                    "title": "Energy Consumption (Wh)",
                },
                "y": {
                    "field": "${field:accuracy}",
                    "type": "quantitative",
                    "title": "Accuracy (%)",
                },
                "color": {
                    "field": "${field:name}",
                    "type": "nominal",
                    "legend": {"title": "Run"},
                },
                "tooltip": [
                    {"field": "${field:name}", "type": "nominal", "title": "Run"},
                    {
                        "field": "${field:energy}",
                        "type": "quantitative",
                        "title": "Energy (Wh)",
                        "format": ".2f",
                    },
                    {
                        "field": "${field:accuracy}",
                        "type": "quantitative",
                        "title": "Accuracy (%)",
                        "format": ".2f",
                    },
                ],
            },
        },
    ],
}


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
            PARETO_NAME_KEY: run.name,
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


def ensure_chart_preset(
    entity: str,
    preset_name: str = DEFAULT_CHART_PRESET_NAME,
    api: Optional[wandb.Api] = None,
) -> str:
    """Register (or re-use) the Pareto frontier Vega-Lite preset on wandb.

    Returns the fully-qualified chart id (``entity/preset_name``) that can be
    passed as ``chart_name`` to :class:`wr.CustomChart`.

    Note: wandb does not expose an update or delete API for chart presets.
    If the preset already exists (HTTP 409), it is reused as-is.  To update
    the Vega spec, delete the old preset manually via the wandb UI
    (Entity → Chart Presets) and re-run this function.
    """
    api = api or wandb.Api()
    chart_id = f"{entity}/{preset_name}"

    try:
        api.create_custom_chart(
            entity=entity,
            name=preset_name,
            display_name="Energy vs Accuracy — Pareto Frontier",
            spec_type="vega2",
            access="private",
            spec=PARETO_VEGA_SPEC,
        )
        _logger.info("Created chart preset: %s", chart_id)
    except Exception as exc:
        _logger.info(
            "Chart preset %s already exists (reusing): %s",
            chart_id,
            exc,
        )
    return chart_id


def ensure_project_scatter_panel(
    entity: str,
    project: str,
    panel_title: str = DEFAULT_PANEL_TITLE,
    section_name: str = DEFAULT_SECTION_NAME,
    workspace_name: str = "Pareto Frontier",
    preset_name: str = DEFAULT_CHART_PRESET_NAME,
) -> str:
    """Upsert a project-level custom chart panel with the Pareto frontier.

    Creates (or replaces, if it already exists by name) a saved workspace
    view containing a layered Vega-Lite panel (scatter + black dashed Pareto
    frontier line).  The panel auto-updates as new runs appear because it
    reads ``pareto/*`` summary fields from the active run set.

    Returns the workspace URL.
    """
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    chart_id = f"{entity}/{preset_name}"

    pareto_chart = wr.CustomChart(
        query={"summary": {"keys": [
            PARETO_ENERGY_WH_KEY,
            PARETO_ACCURACY_KEY,
            PARETO_IS_OPTIMAL_KEY,
            PARETO_NAME_KEY,
        ]}},
        chart_name=chart_id,
        chart_fields={
            "energy": PARETO_ENERGY_WH_KEY,
            "accuracy": PARETO_ACCURACY_KEY,
            "is_pareto": PARETO_IS_OPTIMAL_KEY,
            "name": PARETO_NAME_KEY,
        },
        chart_strings={"title": panel_title},
    )

    workspace = ws.Workspace(
        name=workspace_name,
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name=section_name,
                panels=[pareto_chart],
                is_open=True,
            ),
        ],
    )

    saved = workspace.save()
    url = getattr(saved, "url", None) or getattr(workspace, "url", "")
    _logger.info("Workspace upserted: %s", url)
    return url
