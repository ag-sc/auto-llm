"""Fetch emissions/eval summaries from W&B into ``emissions.json``.

Queries a wandb project (default ``llm4kmu/open-medical-llm-energy``) via the
public API, keeps only finished runs whose name ends with ``-energy``, and
writes each run's ``summary`` dict — plus the run id/name — to a JSON file
matching the existing ``{"runs": [ ... ]}`` schema at the repo root.

Usage::

    python scripts/fetch_wandb_emissions.py
    python scripts/fetch_wandb_emissions.py \
        --entity llm4kmu --project open-medical-llm-energy \
        --output emissions.json

Authentication uses the standard wandb mechanism (``WANDB_API_KEY`` env var or
``~/.netrc``).
"""

import argparse
import json
import logging
import sys
from typing import Any

import wandb

_logger = logging.getLogger("fetch_wandb_emissions")


def _to_plain(value: Any) -> Any:
    """Recursively convert wandb summary values into JSON-serializable objects."""
    # wandb's SummarySubDict / SummaryDict behave like mappings.
    if hasattr(value, "items") and not isinstance(value, dict):
        try:
            value = dict(value)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fetch_runs(
    entity: str,
    project: str,
    suffix: str,
    state: str = "finished",
) -> list[dict]:
    api = wandb.Api()
    path = f"{entity}/{project}"
    _logger.info("Fetching runs from %s", path)
    records: list[dict] = []
    for run in api.runs(path):
        if state and run.state != state:
            continue
        if suffix and not run.name.endswith(suffix):
            continue
        summary = _to_plain(run.summary)
        if not isinstance(summary, dict):
            summary = {}
        record = {"run_id": run.id, "run_name": run.name, **summary}
        records.append(record)
        _logger.info("  kept %s (%s)", run.name, run.id)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch finished *-energy runs from a wandb project and dump their "
            "summaries to emissions.json."
        )
    )
    parser.add_argument("--entity", default="llm4kmu", help="wandb entity (team/user).")
    parser.add_argument(
        "--project", default="open-medical-llm-energy", help="wandb project name."
    )
    parser.add_argument(
        "--suffix",
        default="-energy",
        help="Only keep runs whose name ends with this suffix (use '' to disable).",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Only keep runs with this wandb state (use '' to disable).",
    )
    parser.add_argument(
        "--output",
        default="emissions.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    records = fetch_runs(
        entity=args.entity,
        project=args.project,
        suffix=args.suffix,
        state=args.state,
    )

    if not records:
        _logger.warning("No runs matched filters; writing empty runs array.")

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"runs": records}, fh, indent=2, default=str)

    _logger.info("Wrote %d run(s) to %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
