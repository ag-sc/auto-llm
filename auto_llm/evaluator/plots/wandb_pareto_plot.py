"""Project-level Pareto frontier plots on wandb (multi-panel).

Backfills ``pareto/<label>/*`` fields into every energy run's summary, then
upserts a workspace containing one custom-chart panel per label.  Labels are
namespaced: ``overall``, ``group/<group_name>``, ``task/<task_name>`` — yielding
1 overall + 3 group + 9 per-task Pareto panels for the Open Medical LLM
benchmark.  All panels share a single Vega-Lite chart preset (scatter + black
dashed frontier); only the ``chart_fields`` mapping differs per panel.

Historical runs (logged before ``eval/task/*`` / ``eval/group/*`` keys existed)
are handled by a fallback that reads lm-eval's native per-task keys
(``<task_name>/<metric>[,<filter>]``) and synthesizes the missing scores.
Because lm-eval-harness logs its per-task metrics to its **own** wandb run
(not the energy one), the fallback first looks up an lm-eval *companion*
run in the same project — matched by name (exact, then stripped of a
``-energy`` suffix) — and merges its summary under the energy run's so the
resolver can read the per-task keys.

Typical usage (see also ``scripts/wandb_pareto_plot.py``)::

    specs = build_panels_spec()
    for spec in specs:
        backfill_pareto_flags(
            entity="llm4kmu", project="open-medical-llm-energy",
            label=spec["label"], score_key=spec["score_key"],
            score_resolver=spec["score_resolver"],
        )
    ensure_chart_preset(entity="llm4kmu")
    ensure_project_scatter_panels(
        entity="llm4kmu", project="open-medical-llm-energy",
        panel_specs=[(s["label"], s["title"], s["section"]) for s in specs],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import wandb

from auto_llm.evaluator.plots.pareto_plot import compute_pareto_indices
from auto_llm.evaluator.utils import (
    TASK_DISPLAY_NAMES,
    TASK_GROUP_DISPLAY_NAMES,
    TASK_GROUPS,
)

_logger = logging.getLogger(__name__)


DEFAULT_ENERGY_KEY = "emissions/actual_energy_consumed_kWh"
DEFAULT_SCORE_KEY = "eval/avg_score"
DEFAULT_TAG = "energy-profiling"
DEFAULT_PANEL_TITLE = "Energy vs Accuracy — Pareto Frontier"
DEFAULT_SECTION_NAME = "Pareto Frontier"

PARETO_KEY_PREFIX = "pareto"

# Preset name is defined after PARETO_VEGA_SPEC; see _make_preset_name below.


def pareto_keys(label: str) -> Dict[str, str]:
    """Return the six ``pareto/<label>/*`` summary key names for a panel.

    ``label`` is inserted verbatim into the key path and may itself contain
    slashes (e.g. ``"group/medical_boards"`` → ``"pareto/group/medical_boards/energy_wh"``).
    """
    prefix = f"{PARETO_KEY_PREFIX}/{label}"
    return {
        "energy_wh": f"{prefix}/energy_wh",
        "accuracy_pct": f"{prefix}/accuracy_pct",
        "is_optimal": f"{prefix}/is_optimal",
        "rank": f"{prefix}/rank",
        "name": f"{prefix}/name",
        "variant": f"{prefix}/variant",
        "dataset": f"{prefix}/dataset",
    }


def _classify_variant(run_name: str) -> str:
    """Return one of {"pt", "it", "sft", "qlora"} from a run name.

    Naming convention from this project's eval configs:
      - ``sft-medqa-<model>-energy``          → ``sft``
      - ``sft-medqa-<model>-qlora-energy``    → ``qlora``
      - ``pre-<model>-it-energy``             → ``it``
      - ``pre-<model>-energy`` (incl. quant)  → ``pt``
    """
    name = run_name.lower()
    if name.startswith("sft-"):
        return "qlora" if "qlora" in name else "sft"
    stem = name[: -len("-energy")] if name.endswith("-energy") else name
    if stem.endswith("-it") or "-it-" in stem:
        return "it"
    return "pt"


def _classify_dataset(run_name: str) -> str:
    """Return "mixed" for openmedicalLLM_mixed SFT runs, else "other".

    Mixed-dataset SFT runs are named ``sft-openmedicalLLM_mixed-<model>[-qlora]``;
    this orthogonal flag drives the black outline ring in the scatter layer so
    they stand apart from the single-dataset (medqa / medmcqa) SFT runs that
    share the same shape and per-run color.
    """
    return "mixed" if "openmedicalllm_mixed" in run_name.lower() else "other"

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
                "size": 120,
                "opacity": 0.85,
                "strokeWidth": 2.5,
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
                "shape": {
                    "field": "${field:variant}",
                    "type": "nominal",
                    "scale": {
                        "domain": ["pt", "it", "sft", "qlora"],
                        "range": ["circle", "square", "triangle-up", "diamond"],
                    },
                    "legend": {"title": "Variant"},
                },
                "stroke": {
                    "field": "${field:dataset}",
                    "type": "nominal",
                    "scale": {
                        "domain": ["mixed", "other"],
                        "range": ["black", "transparent"],
                    },
                    "legend": {"title": "Dataset"},
                },
                "tooltip": [
                    {"field": "${field:name}", "type": "nominal", "title": "Run"},
                    {"field": "${field:variant}", "type": "nominal", "title": "Variant"},
                    {"field": "${field:dataset}", "type": "nominal", "title": "Dataset"},
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


def _make_preset_name(base: str = "pareto-frontier") -> str:
    """Suffix the preset name with a short hash of ``PARETO_VEGA_SPEC``.

    wandb has no public update API for chart presets — ``api.create_custom_chart``
    returns HTTP 409 if the name already exists, and the call site silently
    swallows that error. Hashing the spec into the name means any change to the
    Vega spec auto-busts the cache: the new spec lands under a new preset name,
    panels reference the new name, no manual UI step required.
    """
    digest = hashlib.sha1(
        json.dumps(PARETO_VEGA_SPEC, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{base}-{digest}"


DEFAULT_CHART_PRESET_NAME = _make_preset_name()


@dataclass
class ParetoPoint:
    name: str
    run_id: str
    energy_wh: float
    accuracy_pct: float
    is_pareto: bool = False
    rank: int = -1
    variant: str = "pt"
    dataset: str = "other"


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


def _resolve_task_score(summary: Any, task_name: str) -> Optional[float]:
    """Return a task's primary score, with fallback to lm-eval native keys.

    Priority:
    1. ``eval/task/<task_name>`` — the canonical key written by Phase 1.
    2. ``<task_name>/<metric>[,<filter>]`` — lm-eval-harness's own wandb
       integration.  Scans the summary for keys starting with ``<task_name>/``,
       skips ``alias`` and ``*stderr*`` metrics, picks the first numeric value
       alphabetically by base metric name (so ``acc`` beats ``acc_norm``).
    """
    canonical = _get_summary_value(summary, f"eval/task/{task_name}")
    if canonical is not None:
        return canonical

    prefix = f"{task_name}/"
    candidates: List[Tuple[str, float]] = []
    try:
        items = dict(summary).items()
    except Exception:
        return None
    for key, value in items:
        if not key.startswith(prefix):
            continue
        metric_part = key[len(prefix):]
        base_metric = metric_part.split(",")[0]
        if base_metric == "alias" or "stderr" in base_metric:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        candidates.append((base_metric, float(value)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p[0])
    return candidates[0][1]


def _resolve_group_score(summary: Any, group_name: str) -> Optional[float]:
    """Return a group's average score, falling back to member-task resolutions."""
    canonical = _get_summary_value(summary, f"eval/group/{group_name}")
    if canonical is not None:
        return canonical

    members = TASK_GROUPS.get(group_name, [])
    scores: List[float] = []
    for task in members:
        score = _resolve_task_score(summary, task)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def make_task_resolver(task_name: str) -> Callable[[Any], Optional[float]]:
    return lambda summary: _resolve_task_score(summary, task_name)


def make_group_resolver(group_name: str) -> Callable[[Any], Optional[float]]:
    return lambda summary: _resolve_group_score(summary, group_name)


def extract_pareto_points(
    runs: Iterable[Any],
    energy_key: str = DEFAULT_ENERGY_KEY,
    score_key: str = DEFAULT_SCORE_KEY,
    score_scale: float = 100.0,
    score_resolver: Optional[Callable[[Any], Optional[float]]] = None,
    companion_map: Optional[Dict[str, Any]] = None,
) -> List[ParetoPoint]:
    """Pull ``(energy_wh, accuracy_pct)`` from each run and mark Pareto points.

    Energy is always read from the energy run's own summary.  The score is
    read from a *merged* view of the energy summary overlaid on the lm-eval
    companion summary (if one exists in ``companion_map``) — primary wins
    on key conflict — so resolvers can see lm-eval's native per-task keys
    even when they live on a sibling run.
    """
    companion_map = companion_map or {}
    points: List[ParetoPoint] = []
    for run in runs:
        primary_summary = run.summary
        energy_kwh = _get_summary_value(primary_summary, energy_key)

        companion_summary = find_companion(companion_map, run.name)
        merged_summary = _merge_summaries(primary_summary, companion_summary)
        if score_resolver is not None:
            score = score_resolver(merged_summary)
        else:
            score = _get_summary_value(merged_summary, score_key)

        if energy_kwh is None or score is None:
            _logger.debug(
                "Skipping run %s: missing %s or score", run.name, energy_key
            )
            continue
        points.append(
            ParetoPoint(
                name=run.name,
                run_id=run.id,
                energy_wh=energy_kwh * 1000.0,
                accuracy_pct=score * score_scale,
                variant=_classify_variant(run.name),
                dataset=_classify_dataset(run.name),
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


def build_companion_map(
    all_runs: Iterable[Any],
    energy_tag: Optional[str] = DEFAULT_TAG,
) -> Dict[str, Any]:
    """Index non-energy runs by name so lm-eval siblings can be looked up.

    lm-eval-harness logs per-task metrics (``<task>/<metric>[,<filter>]``) to
    its own wandb run, not the ``WandbEnergyLogger`` run.  This helper walks
    the full project and keeps only runs **without** the ``energy_tag``,
    keyed by run name.  ``find_companion`` then matches energy runs to their
    lm-eval counterparts.
    """
    companions: Dict[str, Any] = {}
    for run in all_runs:
        tags = run.tags or []
        if energy_tag and energy_tag in tags:
            continue
        companions[run.name] = run.summary
    return companions


def find_companion(
    companion_map: Dict[str, Any],
    energy_run_name: str,
) -> Optional[Any]:
    """Return the lm-eval companion summary for an energy run, or None.

    Tries exact name match first (both runs share ``wandb_args.name`` from
    the YAML), then strips a trailing ``-energy`` suffix as a fallback for
    configs that use the suffix only on the energy run.
    """
    if energy_run_name in companion_map:
        return companion_map[energy_run_name]
    if energy_run_name.endswith("-energy"):
        return companion_map.get(energy_run_name[: -len("-energy")])
    return None


def _merge_summaries(primary: Any, companion: Optional[Any]) -> Any:
    """Overlay ``primary`` on top of ``companion`` (primary wins on conflict).

    Returns ``primary`` unchanged if ``companion`` is None or either value
    cannot be coerced to a dict — score resolution then falls back to the
    primary-only path.
    """
    if companion is None:
        return primary
    try:
        merged = dict(companion)
    except (TypeError, ValueError):
        return primary
    try:
        for key, value in dict(primary).items():
            merged[key] = value
    except (TypeError, ValueError):
        return primary
    return merged


def backfill_pareto_flags(
    entity: str,
    project: str,
    label: str = "overall",
    energy_key: str = DEFAULT_ENERGY_KEY,
    score_key: str = DEFAULT_SCORE_KEY,
    score_resolver: Optional[Callable[[Any], Optional[float]]] = None,
    score_scale: float = 100.0,
    tag: Optional[str] = DEFAULT_TAG,
    dry_run: bool = False,
    api: Optional[wandb.Api] = None,
    runs: Optional[List[Any]] = None,
    companion_map: Optional[Dict[str, Any]] = None,
) -> List[ParetoPoint]:
    """Write ``pareto/<label>/*`` fields into each qualifying run's summary.

    Returns the list of points (with is_pareto / rank populated). When
    ``dry_run`` is set, nothing is persisted.  Pass an already-fetched ``runs``
    list and ``companion_map`` to avoid re-hitting the wandb API when looping
    over many labels.
    """
    if runs is None:
        api = api or wandb.Api()
        all_runs = list(api.runs(f"{entity}/{project}"))
        runs = _filter_runs_by_tag(all_runs, tag)
        if companion_map is None:
            companion_map = build_companion_map(all_runs, energy_tag=tag)
    if not runs:
        _logger.warning(
            "No runs found in %s/%s with tag %r", entity, project, tag
        )
        return []

    points = extract_pareto_points(
        runs,
        energy_key=energy_key,
        score_key=score_key,
        score_scale=score_scale,
        score_resolver=score_resolver,
        companion_map=companion_map,
    )
    if not points:
        _logger.warning(
            "[%s] No runs expose both %s and a score via %s",
            label,
            energy_key,
            "resolver" if score_resolver else score_key,
        )
        return []

    keys = pareto_keys(label)
    runs_by_id = {r.id: r for r in runs}
    written = 0
    for point in points:
        run = runs_by_id[point.run_id]
        update: Dict[str, Any] = {
            keys["energy_wh"]: point.energy_wh,
            keys["accuracy_pct"]: point.accuracy_pct,
            keys["is_optimal"]: bool(point.is_pareto),
            keys["rank"]: int(point.rank),
            keys["name"]: run.name,
            keys["variant"]: point.variant,
            keys["dataset"]: point.dataset,
        }
        if dry_run:
            flag = "pareto" if point.is_pareto else "       "
            print(
                f"  [{label}] [{flag}] {run.name}: "
                f"energy={point.energy_wh:.2f} Wh, "
                f"acc={point.accuracy_pct:.2f}%, rank={point.rank}"
            )
            continue
        run.summary.update(update)
        run.update()
        written += 1

    if not dry_run:
        _logger.info(
            "Backfilled pareto/%s/* on %d runs in %s/%s",
            label,
            written,
            entity,
            project,
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


def build_panels_spec() -> List[Dict[str, Any]]:
    """Return the 13 panel specs: 1 overall + 3 groups + 9 tasks.

    Each spec dict carries:
      - ``label``: namespace inserted into ``pareto/<label>/*`` summary keys
      - ``section``: workspace section the panel belongs to
      - ``title``: human-readable chart title
      - ``score_key``: canonical (Phase-1) wandb key for this score
      - ``score_resolver``: callable that reads a run's summary and returns
        the score (checks canonical key first, then falls back to lm-eval's
        native ``<task>/<metric>`` keys for historical runs)

    The resolver for the ``overall`` panel is ``None`` because
    ``eval/avg_score`` already exists on every historical run.
    """
    specs: List[Dict[str, Any]] = []

    specs.append({
        "label": "overall",
        "section": "Overall",
        "title": "Overall — Energy vs Accuracy",
        "score_key": "eval/avg_score",
        "score_resolver": None,
    })

    for group_name in TASK_GROUPS:
        display = TASK_GROUP_DISPLAY_NAMES.get(group_name, group_name)
        specs.append({
            "label": f"group/{group_name}",
            "section": "By Group",
            "title": f"{display} — Energy vs Accuracy",
            "score_key": f"eval/group/{group_name}",
            "score_resolver": make_group_resolver(group_name),
        })

    for task_name in (t for tasks in TASK_GROUPS.values() for t in tasks):
        display = TASK_DISPLAY_NAMES.get(task_name, task_name)
        specs.append({
            "label": f"task/{task_name}",
            "section": "By Task",
            "title": f"{display} — Energy vs Accuracy",
            "score_key": f"eval/task/{task_name}",
            "score_resolver": make_task_resolver(task_name),
        })

    return specs


def ensure_project_scatter_panels(
    entity: str,
    project: str,
    panel_specs: List[Tuple[str, str, str]],
    workspace_name: str = "Pareto Frontier",
    preset_name: str = DEFAULT_CHART_PRESET_NAME,
) -> str:
    """Upsert a project workspace with one Pareto panel per spec.

    ``panel_specs`` is a list of ``(label, title, section_name)`` tuples — one
    per panel.  Panels sharing a ``section_name`` are grouped into the same
    workspace section, preserving insertion order.  All panels reference the
    single Vega-Lite preset registered by ``ensure_chart_preset``; only the
    ``chart_fields`` mapping differs per panel (pointing at the namespaced
    ``pareto/<label>/*`` summary keys).

    Returns the workspace URL.
    """
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    chart_id = f"{entity}/{preset_name}"

    sections: Dict[str, List[Any]] = {}
    for label, title, section_name in panel_specs:
        keys = pareto_keys(label)
        panel = wr.CustomChart(
            query={"summary": {"keys": [
                keys["energy_wh"],
                keys["accuracy_pct"],
                keys["is_optimal"],
                keys["name"],
                keys["variant"],
                keys["dataset"],
            ]}},
            chart_name=chart_id,
            chart_fields={
                "energy": keys["energy_wh"],
                "accuracy": keys["accuracy_pct"],
                "is_pareto": keys["is_optimal"],
                "name": keys["name"],
                "variant": keys["variant"],
                "dataset": keys["dataset"],
            },
            chart_strings={"title": title},
        )
        sections.setdefault(section_name, []).append(panel)

    workspace = ws.Workspace(
        name=workspace_name,
        entity=entity,
        project=project,
        sections=[
            ws.Section(name=name, panels=panels, is_open=True)
            for name, panels in sections.items()
        ],
    )

    saved = workspace.save()
    url = getattr(saved, "url", None) or getattr(workspace, "url", "")
    _logger.info("Workspace upserted: %s", url)
    return url


class ParetoRefreshError(RuntimeError):
    """Raised when ``refresh_pareto_workspace`` cannot produce any Pareto data."""


def refresh_pareto_workspace(
    entity: str,
    project: str,
    *,
    energy_key: str = DEFAULT_ENERGY_KEY,
    score_scale: float = 100.0,
    tag: Optional[str] = DEFAULT_TAG,
    workspace_name: str = "Pareto Frontier",
    dry_run: bool = False,
    skip_backfill: bool = False,
    skip_panel: bool = False,
    skip_preset: bool = False,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Run the full backfill + chart-preset + workspace upsert pipeline.

    Library entry point used by ``scripts/wandb_pareto_plot.py`` and by the
    in-process auto-refresh hook in ``auto_llm.evaluator.run`` when
    ``auto_pareto.enabled`` is set in the eval YAML config.

    Returns the workspace URL on a successful upsert, ``None`` for dry-run or
    when the panel step is skipped. Raises :class:`ParetoRefreshError` when
    no qualifying runs exist or no panel produced any points.
    """
    specs = build_panels_spec()

    if not skip_backfill:
        api = wandb.Api(api_key=api_key)
        all_runs = list(api.runs(f"{entity}/{project}"))
        runs = _filter_runs_by_tag(all_runs, tag)
        if not runs:
            raise ParetoRefreshError(
                f"No runs found in {entity}/{project} with tag {tag!r}."
            )

        companion_map = build_companion_map(all_runs, energy_tag=tag)
        _logger.info(
            "Processing %d panel(s) over %d run(s); %d lm-eval companion(s) "
            "available for fallback.",
            len(specs),
            len(runs),
            len(companion_map),
        )
        any_written = False
        for spec in specs:
            points = backfill_pareto_flags(
                entity=entity,
                project=project,
                label=spec["label"],
                energy_key=energy_key,
                score_key=spec["score_key"],
                score_resolver=spec["score_resolver"],
                score_scale=score_scale,
                tag=tag,
                dry_run=dry_run,
                api=api,
                runs=runs,
                companion_map=companion_map,
            )
            if points:
                any_written = True
            n_pareto = sum(1 for p in points if p.is_pareto)
            suffix = " (dry-run)" if dry_run else ""
            _logger.info(
                "[%-40s] %d run(s), %d on frontier%s",
                spec["label"],
                len(points),
                n_pareto,
                suffix,
            )

        if not any_written:
            raise ParetoRefreshError("No panel produced any points.")

    if dry_run or skip_panel:
        return None

    if not skip_preset:
        ensure_chart_preset(entity=entity, api=wandb.Api(api_key=api_key))

    panel_specs = [(s["label"], s["title"], s["section"]) for s in specs]
    return ensure_project_scatter_panels(
        entity=entity,
        project=project,
        panel_specs=panel_specs,
        workspace_name=workspace_name,
    )
