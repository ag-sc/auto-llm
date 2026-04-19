"""Backfill pareto/* summary fields and upsert a multi-panel Pareto workspace.

Loops over 13 score labels (1 overall + 3 source groups + 9 tasks) defined by
``build_panels_spec`` in ``auto_llm.evaluator.plots.wandb_pareto_plot``:

  * Writes ``pareto/<label>/{energy_wh,accuracy_pct,is_optimal,rank,name}``
    into each qualifying run's summary.  For historical runs missing the
    canonical ``eval/task/*`` / ``eval/group/*`` keys, a fallback resolver
    reads lm-eval's native ``<task>/<metric>`` keys and synthesizes the score.
  * Registers (or reuses) the shared Vega-Lite Pareto chart preset.
  * Upserts a workspace view with three sections ("Overall", "By Group",
    "By Task") containing one custom-chart panel per label.

Usage::

    python scripts/wandb_pareto_plot.py --entity llm4kmu --project open-medical-llm-energy
    python scripts/wandb_pareto_plot.py --entity llm4kmu --project open-medical-llm-energy --dry-run
"""

import argparse
import logging
import sys

import wandb

from auto_llm.evaluator.plots.wandb_pareto_plot import (
    DEFAULT_ENERGY_KEY,
    DEFAULT_TAG,
    _filter_runs_by_tag,
    backfill_pareto_flags,
    build_panels_spec,
    ensure_chart_preset,
    ensure_project_scatter_panels,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the multi-panel Pareto workspace on a wandb project."
    )
    parser.add_argument("--entity", required=True, help="wandb entity (team/user)")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument(
        "--energy-key",
        default=DEFAULT_ENERGY_KEY,
        help="Run-summary key holding the energy metric (kWh).",
    )
    parser.add_argument(
        "--score-scale",
        type=float,
        default=100.0,
        help="Multiplier applied to each score to reach a %% scale.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Only consider runs carrying this tag (set to '' to disable).",
    )
    parser.add_argument(
        "--workspace-name",
        default="Pareto Frontier",
        help="Name of the saved workspace view upserted on wandb.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-label frontier summaries without writing anything.",
    )
    parser.add_argument(
        "--skip-backfill", action="store_true", help="Skip the summary write step."
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip the workspace panel upsert step.",
    )
    parser.add_argument(
        "--skip-preset",
        action="store_true",
        help="Skip the chart preset creation step (use if already registered).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    tag = args.tag or None
    specs = build_panels_spec()

    if not args.skip_backfill:
        api = wandb.Api()
        all_runs = list(api.runs(f"{args.entity}/{args.project}"))
        runs = _filter_runs_by_tag(all_runs, tag)
        if not runs:
            print(
                f"No runs found in {args.entity}/{args.project} with tag "
                f"{tag!r}; aborting."
            )
            return 1

        print(f"Processing {len(specs)} panel(s) over {len(runs)} run(s)...\n")
        any_written = False
        for spec in specs:
            points = backfill_pareto_flags(
                entity=args.entity,
                project=args.project,
                label=spec["label"],
                energy_key=args.energy_key,
                score_key=spec["score_key"],
                score_resolver=spec["score_resolver"],
                score_scale=args.score_scale,
                tag=tag,
                dry_run=args.dry_run,
                runs=runs,
            )
            if points:
                any_written = True
            n_pareto = sum(1 for p in points if p.is_pareto)
            suffix = " (dry-run)" if args.dry_run else ""
            print(
                f"[{spec['label']:<40}] {len(points)} run(s), "
                f"{n_pareto} on frontier{suffix}"
            )

        if not any_written:
            print("\nNo panel produced any points; aborting.")
            return 1

    if args.dry_run or args.skip_panel:
        return 0

    if not args.skip_preset:
        ensure_chart_preset(entity=args.entity)

    panel_specs = [(s["label"], s["title"], s["section"]) for s in specs]
    url = ensure_project_scatter_panels(
        entity=args.entity,
        project=args.project,
        panel_specs=panel_specs,
        workspace_name=args.workspace_name,
    )
    if url:
        print(f"\nWorkspace view: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
