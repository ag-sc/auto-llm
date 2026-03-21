import csv
import logging
from pathlib import Path
from typing import Dict, Optional

from auto_llm.constants import EMISSIONS_CSV_FILENAME

logger = logging.getLogger(__name__)


def read_last_emissions_row(output_dir: str) -> Optional[Dict[str, str]]:
    """Return the last row of the CodeCarbon ``emissions.csv`` as a dict.

    Args:
        output_dir: Directory containing the CSV file.

    Returns:
        A ``{column: value}`` dict for the last row, or ``None`` when the
        file does not exist, is empty, or cannot be parsed.
    """
    csv_path = Path(output_dir) / EMISSIONS_CSV_FILENAME
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        return rows[-1]
    except Exception as exc:
        logger.warning("Failed to parse emissions CSV: %s", exc)
        return None


def parse_wandb_args(raw: Optional[str]) -> Dict[str, str]:
    """Parse a ``wandb_args`` string into a dict.

    The expected format is the comma-separated ``key=value`` convention used
    by lm-eval-harness, e.g. ``"project=my-project,name=my-run"``.

    Returns an empty dict when *raw* is ``None`` or empty.
    """
    if not raw:
        return {}
    result: Dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        result[key.strip()] = value.strip()
    return result
