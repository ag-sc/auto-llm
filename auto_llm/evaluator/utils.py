from collections import defaultdict
from typing import Any, Dict, List, Optional

from lm_eval.__main__ import setup_parser, cli_evaluate
import lm_eval.evaluator as _lm_eval_evaluator
from lm_eval.tasks import TaskManager

from auto_llm.registry.evaluator_registry import LM_EVAL_HARNESS_CUSTOM_TASKS_PATH

# TODO: ideally this should come from the eval config file. But causes a lots of delay.
LM_EVAL_TASK_MANAGER = TaskManager(include_path=LM_EVAL_HARNESS_CUSTOM_TASKS_PATH)


# Open Medical LLM Benchmark — source-based groupings (not defined by the HF
# leaderboard; chosen for multi-level Pareto frontier analysis).
TASK_GROUPS: Dict[str, List[str]] = {
    "medical_boards": ["medqa_4options", "medmcqa"],
    "literature_qa": ["pubmedqa"],
    "mmlu_medical": [
        "mmlu_anatomy_generative",
        "mmlu_clinical_knowledge_generative",
        "mmlu_college_biology_generative",
        "mmlu_college_medicine_generative",
        "mmlu_medical_genetics_generative",
        "mmlu_professional_medicine_generative",
    ],
}

TASK_GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "medical_boards": "Medical Board Exams",
    "literature_qa": "Biomedical Literature QA",
    "mmlu_medical": "MMLU Medical Subsets",
}

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "medqa_4options": "MedQA (USMLE, 4-opt)",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA",
    "mmlu_anatomy_generative": "MMLU Anatomy",
    "mmlu_clinical_knowledge_generative": "MMLU Clinical Knowledge",
    "mmlu_college_biology_generative": "MMLU College Biology",
    "mmlu_college_medicine_generative": "MMLU College Medicine",
    "mmlu_medical_genetics_generative": "MMLU Medical Genetics",
    "mmlu_professional_medicine_generative": "MMLU Professional Medicine",
}


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
    integration under its native key scheme.  This function re-exposes them
    under a uniform ``eval/*`` namespace and adds group/overall averages::

        {
            "eval/avg/acc": 0.71,
            "eval/avg/acc_norm": 0.69,
            "eval/avg/exact_match": 0.76,
            "eval/avg_score": 0.75,
            "eval/task/medmcqa": 0.72,
            "eval/task/pubmedqa": 0.78,
            ...
            "eval/group/medical_boards": 0.70,
            "eval/group/mmlu_medical": 0.74,
            ...
        }

    ``eval/avg/{metric}`` averages each metric across tasks that report it.
    ``eval/avg_score`` averages one primary score per task (the first
    non-stderr metric, matching lm-eval's metric_list ordering).
    ``eval/task/{task_name}`` is the same primary score, per task.
    ``eval/group/{group_name}`` averages the primary scores of the member
    tasks (from ``TASK_GROUPS``) actually present in this run; groups with
    no members present are omitted.

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

    for task_name, score in primary_scores.items():
        wandb_metrics[f"eval/task/{task_name}"] = score

    for group_name, member_tasks in TASK_GROUPS.items():
        present = [primary_scores[t] for t in member_tasks if t in primary_scores]
        if present:
            wandb_metrics[f"eval/group/{group_name}"] = sum(present) / len(present)

    return wandb_metrics
