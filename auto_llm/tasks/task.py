import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type, List

from pydantic import BaseModel

from auto_llm.dto.trainer_run_config import TrainerRunConfig


class Task(BaseModel):
    name: str
    description: str

    sample_trainer_run_config: TrainerRunConfig = None
    sample_evaluator_run_config: TrainerRunConfig = None



class TaskRegistry:
    _registry: Dict[str, Type[Task]] = {}


    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: Type[Task]):
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get_task(cls, name: str) -> Task:
        task_cls = cls._registry.get(name)
        if not task_cls:
            raise ValueError(f"Task '{name}' is not registered.")
        return task_cls()

    @classmethod
    def get_task_names(cls) -> List[str]:
        return list(cls._registry.keys())

    @staticmethod
    def setup_tasks():
        tasks_path = Path(__file__).resolve().parent
        print(f"Loading tasks from {tasks_path}")
        package_name = __package__

        for loader, module_name, is_pkg in pkgutil.walk_packages([str(tasks_path)]):
            full_module_name = f'.{module_name}'

            if full_module_name == ".task":
                continue

            importlib.import_module(full_module_name, package=package_name)
            print(f"Registered task: {full_module_name}")
