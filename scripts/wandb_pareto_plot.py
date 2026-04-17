"""Backfill pareto/* summary fields and upsert a project-level scatter panel.

Reads every ``energy-profiling``-tagged run in a wandb project, computes the
Pareto frontier over ``(emissions/energy_consumed_kWh, eval/avg_score)``, and
writes the per-run ``pareto/*`` fields that drive a project ScatterPlot
panel. Also upserts the workspace view containing that panel.

Usage::

    python scripts/wandb_pareto_plot.py --entity llm4kmu --project open-medical-llm-energy
    python scripts/wandb_pareto_plot.py --entity llm4kmu --project open-medical-llm-energy --dry-run
"""

import argparse
import logging
import sys

from auto_llm.evaluator.plots.wandb_pareto_plot import (
    DEFAULT_ENERGY_KEY,
    DEFAULT_PANEL_TITLE,
    DEFAULT_SCORE_KEY,
    DEFAULT_TAG,
    backfill_pareto_flags,
    ensure_project_scatter_panel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the Pareto frontier on a wandb project."
    )
    parser.add_argument("--entity", required=True, help="wandb entity (team/user)")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument(
        "--energy-key",
        default=DEFAULT_ENERGY_KEY,
        help="Run-summary key holding the energy metric (kWh).",
    )
    parser.add_argument(
        "--score-key",
        default=DEFAULT_SCORE_KEY,
        help="Run-summary key holding the accuracy metric (0-1).",
    )
    parser.add_argument(
        "--score-scale",
        type=float,
        default=100.0,
        help="Multiplier applied to the score to reach a %% scale.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Only consider runs carrying this tag (set to '' to disable).",
    )
    parser.add_argument(
        "--panel-title",
        default=DEFAULT_PANEL_TITLE,
        help="Title of the scatter panel written to the workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frontier and planned writes without persisting anything.",
    )
    parser.add_argument(
        "--skip-backfill", action="store_true", help="Skip the summary write step."
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip the workspace panel upsert step.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    tag = args.tag or None

    if not args.skip_backfill:
        points = backfill_pareto_flags(
            entity=args.entity,
            project=args.project,
            energy_key=args.energy_key,
            score_key=args.score_key,
            score_scale=args.score_scale,
            tag=tag,
            dry_run=args.dry_run,
        )
        if not points:
            print("No eligible runs found; aborting.")
            return 1
        n_pareto = sum(1 for p in points if p.is_pareto)
        print(
            f"\n{len(points)} run(s), {n_pareto} on the frontier"
            + (" (dry-run, nothing written)." if args.dry_run else ".")
        )

    if args.dry_run or args.skip_panel:
        return 0

    url = ensure_project_scatter_panel(
        entity=args.entity,
        project=args.project,
        panel_title=args.panel_title,
    )
    if url:
        print(f"Workspace view: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
