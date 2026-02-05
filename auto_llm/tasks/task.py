from typing import List

from pydantic import BaseModel

from auto_llm.dto.trainer_run_config import TrainerRunConfig


class BaseTask(BaseModel):
    name: str
    description: str

    metrics_list: List[str]
    models_list: List[str]

    sample_trainer_run_config: TrainerRunConfig = None
    sample_evaluator_run_config: TrainerRunConfig = None


class Task:
    base_task: BaseTask
    task_details: str
