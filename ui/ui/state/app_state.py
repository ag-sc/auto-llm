import asyncio
import datetime
import json
import os
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import reflex as rx

from auto_llm.automator.automator import Automator
from auto_llm.configurator.config_executor import SequentialConfigExecutor, TaskSpoolerSequentialConfigExecutor
from auto_llm.configurator.config_generator import ConfiguratorOutput, Priority, TrainEvalRunConfigurator
from auto_llm.estimator.utils import get_gpu_params
from auto_llm.tasks.registry import TASKS

from ..backend import CONFIGS_DIR, OUTPUT_DIR, EVAL_RESULTS_DIR
from ..state.user import User, get_wandb_client

GPU_PARAMS = get_gpu_params()

# Bounded multi-user resource pools to prevent thread exhaustion or API rate limiting
_api_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="wandb_api")
_disk_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="disk_io")


class AppState(rx.State):
    current_tab: str = "settings"
    is_rerouted: bool = False
    is_loading: bool = False

    model_choices: list[str] = []
    model_results: pd.DataFrame = pd.DataFrame()

    # settings tab
    dataset_path: str = ""
    task_category: str = ""
    hardware_type: str = "NVIDIA L40S"
    hardware_count: str = "2"

    # models tab
    selected_models: List[str] = []

    # prompts tab
    instruction_template: str = ""
    input_template: str = ""
    output_template: str = ""

    configurator_outputs: List[ConfiguratorOutput] = []
    configs_path: str = ""
    run_group: str = ""

    start_execution: bool = False

    @rx.event
    def toggle_choice(self, choice: str, checked: bool):
        """Add or remove the choice safely with explicit array copy reassignment."""
        current_models = list(self.selected_models)
        if checked:
            if choice not in current_models:
                current_models.append(choice)
        else:
            if choice in current_models:
                current_models.remove(choice)
        self.selected_models = current_models

    @rx.event
    def reset_state(self):
        if self.is_rerouted:
            self.is_rerouted = False
            return

        self.current_tab = "settings"
        self.is_loading = False
        self.model_choices = []
        self.model_results = pd.DataFrame()

        # settings tab
        self.dataset_path = ""
        self.task_category = ""
        self.hardware_type = "NVIDIA L40S"
        self.hardware_count = "2"

        # models tab
        self.selected_models = []

        # prompts tab
        self.instruction_template = ""
        self.input_template = ""
        self.output_template = ""

        self.configurator_outputs = []
        self.configs_path = ""
        self.run_group = ""
        self.start_execution = False
        self.is_rerouted = False

    @rx.event
    def load_from_json(self, data: dict):
        self.is_rerouted = True
        self.current_tab = "validate"
        self.is_loading = False

        self.model_choices = data.get("model_choices") or []
        self.model_results = pd.DataFrame(data.get("model_results") or [])

        # settings tab
        self.dataset_path = data.get("dataset_path", "")
        self.task_category = data.get("task_category", "")
        self.hardware_type = data.get("hardware_type", "NVIDIA L40S")
        self.hardware_count = data.get("hardware_count", "2")

        # models tab
        self.selected_models = data.get("selected_models") or []

        # prompts tab
        self.instruction_template = data.get("instruction_template", "")
        self.input_template = data.get("input_template", "")
        self.output_template = data.get("output_template", "")

        raw_outputs = data.get("configurator_outputs") or []
        self.configurator_outputs = [ConfiguratorOutput.model_validate(x) for x in raw_outputs]
        self.configs_path = data.get("configs_path", "")
        self.run_group = data.get("run_group", "")
        self.start_execution = True

    @rx.var
    def dataset_options_markdown(self) -> str:
        datasets = Automator.get_datasets()
        prefix = "https://huggingface.co/datasets"
        return "\n".join([f"* {d} [(link)]({prefix}/{d})" for d in datasets])

    @rx.event
    async def handle_submit(self, form_data: dict):
        self.is_loading = True

        if not form_data.get("dataset_path") or not form_data.get("task_category"):
            yield rx.toast.warning("Please fill in all required fields!")
            self.is_loading = False
            return

        loop = asyncio.get_running_loop()
        # Offload blocking database/automator fetch to the bounded disk/heavy executor
        model_names, results_df = await loop.run_in_executor(
            _disk_executor, self.update_models, self.task_category, self.dataset_path, self.hardware_type, int(self.hardware_count)
        )

        self.model_choices = model_names
        self.model_results = results_df
        self.is_loading = False
        self.current_tab = "models"

        configured_task = TASKS.get(self.task_category)
        if configured_task and configured_task.sample_trainer_run_config:
            builder_config = configured_task.sample_trainer_run_config.trainer_data_builder_config
            self.instruction_template = builder_config.instruction_template
            self.input_template = builder_config.input_template
            self.output_template = builder_config.output_template

    def update_models(self, task: str, dataset: str, hardware_type: str, hardware_count: int):
        configured_task = TASKS.get(task)
        automator = Automator(
            task_type=configured_task.name,
            dataset=dataset,
            hardware_type=hardware_type,
            hardware_count=hardware_count,
        )
        df = automator.get_models_df()
        cols = [df.columns[0]] + list(df.columns[2:])
        return automator.model_names, df[cols].round(2)

    @rx.event
    async def handle_models_submit(self, form_data: dict):
        if not self.selected_models:
            yield rx.toast.warning("Please select at least one model!")
            return
        self.current_tab = "prompts"

    @staticmethod
    def generate_configs(
        model_names: List[str],
        task: str,
        dataset_path: str,
        instruction_template: str,
        input_template: str,
        output_template: str,
        configs_path: str,
        output_path: str,
        eval_results_path: str,
        project_name: str,
    ) -> List[ConfiguratorOutput]:
        configurator = TrainEvalRunConfigurator(
            model_names=model_names,
            task=task,
            dataset_path=dataset_path,
            configs_path=configs_path,
            output_path=output_path,
            eval_results_path=eval_results_path,
            instruction_template=instruction_template,
            input_template=input_template,
            output_template=output_template,
            entity_name=project_name,
        )
        return configurator.generate()

    @rx.event
    async def handle_prompts_submit(self, form_data: dict):
        if not all(
            [
                form_data.get("instruction_template"),
                form_data.get("input_template"),
                form_data.get("output_template"),
            ]
        ):
            yield rx.toast.warning("Please fill in all template fields!")
            return

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)
        entity = client.get_entity()

        configs_path = f"{CONFIGS_DIR}/{user_state.username}/{timestamp_str}_configs"
        output_path = f"{OUTPUT_DIR}/{user_state.username}/{timestamp_str}_sft_models"
        eval_results_path = f"{EVAL_RESULTS_DIR}/{user_state.username}/{timestamp_str}_eval_results"
        configs_group_path = f"{configs_path}/run_group.json"

        loop = asyncio.get_running_loop()
        configurator_outputs = await loop.run_in_executor(
            _disk_executor,
            self.generate_configs,
            self.selected_models,
            self.task_category,
            self.dataset_path,
            self.instruction_template,
            self.input_template,
            self.output_template,
            configs_path,
            output_path,
            eval_results_path,
            entity,
        )

        sorted_outputs = []
        for p in [Priority.PRIORITY_ONE, Priority.PRIORITY_TWO, Priority.PRIORITY_THREE]:
            sorted_outputs.extend([co for co in configurator_outputs if co.priority == p])

        self.configurator_outputs = sorted_outputs

        def _read_group_and_save(group_path, state_data, username):
            os.makedirs(os.path.dirname(group_path), exist_ok=True)
            with open(group_path, "r") as f:
                data = json.load(f)

            with open(f"{configs_path}/configure_state.json", "w+") as f:
                json.dump(state_data, f, indent=4)
            with open(f"{configs_path}/settings.json", "w+") as f:
                json.dump({"username": username, "timestamp": timestamp_str}, f, indent=4)
            return data["run_group"]

        try:
            serializable_state = self._get_serializable_dict()
            self.run_group = await loop.run_in_executor(_disk_executor, _read_group_and_save, configs_group_path, serializable_state, user_state.username)
            self.current_tab = "validate"
        except Exception as e:
            yield rx.toast.warning(f"Failed to process and save group configs: {str(e)}")

    @rx.event
    async def handle_validation_submit_pre(self, form_data: dict):
        if not self.configurator_outputs:
            yield rx.toast.warning("No configurations found!")
            return
        self.current_tab = "validate"

    @rx.event
    async def handle_validation_submit(self, form_data: dict):
        if not self.configurator_outputs:
            yield rx.toast.warning("No configurations found!")
            return

        self.current_tab = "validate"

        if not self.start_execution:
            self.start_execution = True
            # job_id_to_attach = os.getenv("JOB_ID_TO_ATTACH", None)
            # if not job_id_to_attach:
            #     yield rx.toast.error("Executor needs a JobID to attach itself. Please set environment variable `JOB_ID_TO_ATTACH`.")

            executor = TaskSpoolerSequentialConfigExecutor(configurator_outputs=self.configurator_outputs)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_disk_executor, executor.execute)
            yield rx.toast.success("Your jobs are successfully submitted!")

    @rx.event
    async def set_current_tab(self, value: str):
        self.current_tab = value

    def _get_serializable_dict(self):
        """Converts the active session parameters cleanly into a JSON ready dictionary."""
        exclude = ["is_loading", "parent_state", "router_data", "substates", "dirty_vars", "dirty_substates", "router", "is_hydrated"]
        state_dict = {}

        for key in self.get_fields():
            if key in exclude or key.startswith("_"):
                continue

            value = getattr(self, key)
            if isinstance(value, set):
                state_dict[key] = list(value)
            elif isinstance(value, pd.DataFrame):
                state_dict[key] = value.to_dict(orient="records")
            elif key == "configurator_outputs" and value:
                state_dict[key] = [obj.dict() if hasattr(obj, "dict") else str(obj) for obj in value]
            else:
                state_dict[key] = value

        return state_dict
