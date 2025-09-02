from typing import Any, Dict

import yaml
from lm_eval.__main__ import setup_parser
from lm_eval.tasks import TaskManager

# TODO: ideally this should come from the eval config file. But causes a lots of delay.
LM_EVAL_TASK_MANAGER = TaskManager(include_path="config_files/evaluator_configs/tasks")


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


if __name__ == "__main__":
    config_path = "config_files/evaluator_configs/hellaswag_truthful_qa.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lm_eval_args = parse_lm_eval_config(config)
    ...
