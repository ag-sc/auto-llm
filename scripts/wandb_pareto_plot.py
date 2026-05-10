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

from auto_llm.evaluator.plots.wandb_pareto_plot import (
    DEFAULT_ENERGY_KEY,
    DEFAULT_TAG,
    ParetoRefreshError,
    refresh_pareto_workspace,
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

    try:
        url = refresh_pareto_workspace(
            entity=args.entity,
            project=args.project,
            energy_key=args.energy_key,
            score_scale=args.score_scale,
            tag=args.tag or None,
            workspace_name=args.workspace_name,
            dry_run=args.dry_run,
            skip_backfill=args.skip_backfill,
            skip_panel=args.skip_panel,
            skip_preset=args.skip_preset,
        )
    except ParetoRefreshError as exc:
        print(f"{exc} aborting.")
        return 1

    if url:
        print(f"\nWorkspace view: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
