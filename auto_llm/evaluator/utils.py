from collections import defaultdict
from typing import Any, Dict, List, Optional

from lm_eval.__main__ import setup_parser, cli_evaluate
import lm_eval.evaluator as _lm_eval_evaluator
from lm_eval.tasks import TaskManager

from auto_llm.registry.evaluator_registry import LM_EVAL_HARNESS_CUSTOM_TASKS_PATH

# TODO: ideally this should come from the eval config file. But causes a lots of delay.
LM_EVAL_TASK_MANAGER = TaskManager(include_path=LM_EVAL_HARNESS_CUSTOM_TASKS_PATH)


def parse_lm_eval_config(config: Dict[str, Any]):
    # set values from the YAML config
    lm_eval_parser = setup_parser()
    for key, value in config.items():
        lm_eval_parser.set_defaults(**{key: value})
    lm_eval_args = lm_eval_parser.parse_args(
        args=[]
    )  # passing an empty list, otherwise sys.argv[:1] is taken by default
    return lm_eval_args


def get_lm_eval_tasks(lm_eval_args, task_manager: TaskManager = LM_EVAL_TASK_MANAGER):
    task_list = lm_eval_args.tasks.split(",")
    tasks = task_manager.load_task_or_group(task_list=task_list)

    return tasks


def evaluate_and_capture(lm_eval_args) -> Optional[Dict[str, Any]]:
    """Run lm-eval-harness and return the results dict.

    ``cli_evaluate`` does not expose the results of ``simple_evaluate``.
    This function temporarily wraps ``simple_evaluate`` so the results
    dict is captured and returned to the caller while preserving all of
    ``cli_evaluate``'s side-effects (wandb logging, file output, console
    table).
    """
    captured: Dict[str, Any] = {}
    _orig = _lm_eval_evaluator.simple_evaluate

    def _wrapper(*args, **kwargs):
        result = _orig(*args, **kwargs)
        captured["results"] = result
        return result

    _lm_eval_evaluator.simple_evaluate = _wrapper
    try:
        cli_evaluate(args=lm_eval_args)
    finally:
        _lm_eval_evaluator.simple_evaluate = _orig

    return captured.get("results")


def aggregate_eval_scores(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute cross-task metric averages from lm-eval results.

    Per-task metrics are already logged by ``cli_evaluate``'s own wandb
    integration.  This function only computes the averages that lm-eval
    does not provide::

        {
            "eval/avg/acc": 0.71,
            "eval/avg/acc_norm": 0.69,
            "eval/avg/exact_match": 0.76,
            "eval/avg_score": 0.75,
        }

    ``eval/avg/{metric}`` averages each metric across tasks that report it.
    ``eval/avg_score`` averages one primary score per task (the first
    non-stderr metric, matching lm-eval's metric_list ordering).

    Returns an empty dict when no scores can be extracted.
    """
    task_results = eval_results.get("results", {})

    metric_groups: Dict[str, List[float]] = defaultdict(list)
    primary_scores: Dict[str, float] = {}

    for task_name, metrics in task_results.items():
        is_first = True
        for key, value in metrics.items():
            if "stderr" in key or key == "alias":
                continue
            if not isinstance(value, (int, float)):
                continue

            base_metric = key.split(",")[0]
            metric_groups[base_metric].append(float(value))

            if is_first:
                primary_scores[task_name] = float(value)
                is_first = False

    if not primary_scores:
        return {}

    wandb_metrics: Dict[str, float] = {}

    for metric_name, values in metric_groups.items():
        wandb_metrics[f"eval/avg/{metric_name}"] = sum(values) / len(values)

    wandb_metrics["eval/avg_score"] = (
        sum(primary_scores.values()) / len(primary_scores)
    )

    return wandb_metrics
