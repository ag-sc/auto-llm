"""Add eval/avg/* metrics to existing wandb energy-profiling runs.

For each energy run (``{name}-energy``), finds the corresponding eval run
(``{name}``) logged by lm-eval, computes cross-task metric averages, and
writes them to the energy run's summary. This mirrors the logic of
``aggregate_eval_scores`` in ``auto_llm.evaluator.utils`` so that past
runs get the same averaged metrics that future runs will log automatically.

Usage:
    python scripts/wandb_add_eval_averages.py --entity llm4kmu --project open-medical-llm-energy
    python scripts/wandb_add_eval_averages.py --entity llm4kmu --project open-medical-llm-energy --dry-run
"""

import argparse
from collections import defaultdict
from typing import Dict, List

import wandb


ENERGY_RUN_SUFFIX = "-energy"


def aggregate_run_scores(summary: dict) -> Dict[str, float]:
    """Compute cross-task metric averages from a wandb eval run summary.

    Parses lm-eval metric keys (``{task}/{metric}``) from the summary,
    groups by base metric name, and returns the averages.
    """
    task_metrics: Dict[str, Dict[str, float]] = {}

    for key, value in summary.items():
        if not isinstance(value, (int, float)):
            continue
        if "/" not in key:
            continue
        if key.startswith(("_", "emissions/", "eval/")):
            continue
        if "stderr" in key:
            continue

        parts = key.split("/", 1)
        if len(parts) != 2:
            continue

        task_name, metric_name = parts
        if metric_name == "alias":
            continue

        task_metrics.setdefault(task_name, {})[metric_name] = float(value)

    if not task_metrics:
        return {}

    metric_groups: Dict[str, List[float]] = defaultdict(list)
    primary_scores: Dict[str, float] = {}

    for task_name, metrics in task_metrics.items():
        is_first = True
        for metric_key, value in metrics.items():
            base_metric = metric_key.split(",")[0]
            metric_groups[base_metric].append(value)

            if is_first:
                primary_scores[task_name] = value
                is_first = False

    wandb_metrics: Dict[str, float] = {}

    for metric_name, values in metric_groups.items():
        wandb_metrics[f"eval/avg/{metric_name}"] = sum(values) / len(values)

    wandb_metrics["eval/avg_score"] = (
        sum(primary_scores.values()) / len(primary_scores)
    )

    return wandb_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Add eval/avg/* metrics to existing wandb energy runs."
    )
    parser.add_argument("--entity", required=True, help="wandb entity (team/user)")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing to wandb",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    # Index all runs by name
    runs_by_name: Dict[str, wandb.apis.public.runs.Run] = {}
    for run in runs:
        runs_by_name[run.name] = run

    updated = 0
    skipped = 0

    for run_name, run in runs_by_name.items():
        # Only process energy-profiling runs
        if not run_name.endswith(ENERGY_RUN_SUFFIX):
            continue

        # Find the corresponding eval run
        eval_run_name = run_name.removesuffix(ENERGY_RUN_SUFFIX)
        eval_run = runs_by_name.get(eval_run_name)

        if eval_run is None:
            print(f"  {run_name}: no matching eval run '{eval_run_name}', skipping")
            skipped += 1
            continue

        avg_metrics = aggregate_run_scores(dict(eval_run.summary))

        if not avg_metrics:
            print(f"  {run_name}: no task metrics in eval run '{eval_run_name}', skipping")
            skipped += 1
            continue

        avg_score = avg_metrics.get("eval/avg_score", 0)
        n_metrics = len(avg_metrics) - 1

        if args.dry_run:
            print(f"  {run_name}: would write {n_metrics} avg metrics from '{eval_run_name}', avg_score={avg_score:.4f}")
            for k, v in sorted(avg_metrics.items()):
                print(f"    {k}: {v:.4f}")
        else:
            run.summary.update(avg_metrics)
            run.summary.update()
            print(f"  {run_name}: wrote {n_metrics} avg metrics from '{eval_run_name}', avg_score={avg_score:.4f}")
            updated += 1

    print(f"\nDone. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
