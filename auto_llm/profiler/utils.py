from typing import Dict, Optional


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
