import datetime
import json
import os
from typing import Optional, List

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


class AppState(rx.State):
    current_tab = "settings"

    is_rerouted: bool = False

    is_loading: bool = False
    model_choices: list[str] = []
    model_results: pd.DataFrame = pd.DataFrame()

    # settings tab
    dataset_path: Optional[str] = ""
    task_category: Optional[str] = ""
    hardware_type: Optional[str] = "NVIDIA L40S"
    hardware_count: Optional[str] = "2"

    # models tab
    selected_models: Optional[List[str]] = []

    # prompts tab
    instruction_template: Optional[str] = ""
    input_template: Optional[str] = ""
    output_template: Optional[str] = ""

    configurator_outputs: Optional[List[ConfiguratorOutput]] = []
    configs_path: Optional[str] = ""
    run_group: Optional[str] = ""

    start_execution: bool = False

    def toggle_choice(self, choice: str, checked: bool):
        """Add or remove the choice based on checkbox state."""
        if checked:
            self.selected_models.append(choice)
        else:
            self.selected_models.remove(choice)

    @rx.event
    def reset_state(self):
        if self.is_rerouted:
            self.is_rerouted = False
            return

        self.current_tab = "settings"

        self.is_loading: bool = False
        self.model_choices: list[str] = []
        self.model_results: pd.DataFrame = pd.DataFrame()

        # settings tab
        self.dataset_path: Optional[str] = ""
        self.task_category: Optional[str] = ""
        self.hardware_type: Optional[str] = "NVIDIA L40S"
        self.hardware_count: Optional[str] = "2"

        # models tab
        self.selected_models: Optional[List[str]] = []

        # prompts tab
        self.instruction_template: Optional[str] = ""
        self.input_template: Optional[str] = ""
        self.output_template: Optional[str] = ""

        self.configurator_outputs: Optional[List[ConfiguratorOutput]] = []
        self.configs_path: Optional[str] = ""
        self.run_group: Optional[str] = ""

        self.start_execution: bool = False

        self.is_rerouted: bool = False

    def load_from_json(self, data: dict):
        self.is_rerouted = True

        self.current_tab = "validate"

        self.is_loading: bool = False
        self.model_choices: list[str] = data.get("model_choices", None)
        self.model_results: pd.DataFrame = pd.DataFrame(data.get("model_results", None))

        # settings tab
        self.dataset_path: Optional[str] = data.get("dataset_path", None)
        self.task_category: Optional[str] = data.get("task_category", None)
        self.hardware_type: Optional[str] = data.get("hardware_type", None)
        self.hardware_count: Optional[str] = data.get("hardware_count", None)

        # models tab
        self.selected_models: Optional[List[str]] = data.get("selected_models", None)

        # prompts tab
        self.instruction_template: Optional[str] = data.get("instruction_template", None)
        self.input_template: Optional[str] = data.get("input_template", None)
        self.output_template: Optional[str] = data.get("output_template", None)

        self.configurator_outputs: Optional[List[ConfiguratorOutput]] = [ConfiguratorOutput.model_validate(x) for x in data.get("configurator_outputs", None)]
        self.configs_path: Optional[str] = data.get("configs_path", None)
        self.run_group: Optional[str] = data.get("run_group", None)

        self.start_execution: bool = True

    @rx.var
    def dataset_options_markdown(self) -> str:
        datasets = Automator.get_datasets()
        prefix = "https://huggingface.co/datasets"
        return "\n".join([f"* {d} [(link)]({prefix}/{d})" for d in datasets])

    @rx.event
    async def handle_submit(self, form_data: dict):
        self.is_loading = True

        if not all(
            [
                form_data.get("dataset_path"),
                form_data.get("task_category"),
                # form_data.get("hardware_type"),
            ]
        ):
            yield rx.window_alert("Please fill in all required fields!")
            self.is_loading = False
            return

        # Logic to fetch models
        model_names, results_df = self.update_models(task=self.task_category, dataset=self.dataset_path, hardware_type=self.hardware_type, hardware_count=int(self.hardware_count))

        self.model_choices = model_names
        self.model_results = results_df
        self.is_loading = False
        self.current_tab = "models"

        configured_task = TASKS.get(self.task_category)

        self.instruction_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.instruction_template
        self.input_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.input_template
        self.output_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.output_template

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

        df = df.round(2)
        return automator.model_names, df[cols]

    @rx.event
    async def handle_models_submit(self, form_data: dict):
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
    ) -> List[ConfiguratorOutput]:  # type: ignore
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

        configurator_outputs = configurator.generate()
        return configurator_outputs

    @rx.event
    async def handle_prompts_submit(self, form_data: dict):
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)
        entity = client.get_entity()

        configs_path = f"{CONFIGS_DIR}/{user_state.username}/{timestamp_str}_configs/"
        output_path = f"{OUTPUT_DIR}/{user_state.username}/{timestamp_str}_sft_models/"
        eval_results_path = f"{EVAL_RESULTS_DIR}/{user_state.username}/{timestamp_str}_eval_results/"

        configs_group_path = f"{configs_path}/run_group.json"

        configurator_outputs = self.generate_configs(
            model_names=self.selected_models,
            task=self.task_category,
            dataset_path=self.dataset_path,
            instruction_template=self.instruction_template,
            input_template=self.input_template,
            output_template=self.output_template,
            configs_path=configs_path,
            output_path=output_path,
            eval_results_path=eval_results_path,
            project_name=entity,
        )

        sorted_configurator_outputs = []
        for p in [Priority.PRIORITY_ONE, Priority.PRIORITY_TWO, Priority.PRIORITY_THREE]:
            for co in configurator_outputs:
                if co.priority == p:
                    sorted_configurator_outputs.append(co)

        self.configurator_outputs = sorted_configurator_outputs

        with open(configs_group_path, "r") as f:
            data = json.load(f)

        self.run_group = data["run_group"]

        user_state = await self.get_state(User)
        username = user_state.username
        print("saving app state for user", username)
        self.save_app_state(timestamp=timestamp_str, path=configs_path, username=username)

        self.current_tab = "validate"

    def save_app_state(self, timestamp: str, path: str, username: str):
        data = self._get_serializable_dict()
        state_path = f"{path}/configure_state.json"
        with open(state_path, "w+") as f:
            json.dump(data, f, indent=4)

        settings_path = f"{path}/settings.json"

        print("saving settings", username)
        settings = {"username": username, "timestamp": timestamp}
        with open(settings_path, "w+") as f:
            json.dump(settings, f, indent=4)

    @rx.event
    async def handle_validation_submit(self, form_data: dict):
        self.current_tab = "validate"

        if not self.start_execution:
            self.start_execution = True
            # TODO: set executors outside AppState
            job_id_to_attach = os.getenv("JOB_ID_TO_ATTACH", None)
            if not job_id_to_attach:
                rx.toast.error(f"Executor needs a JobID to attach itself. Please set the environment variable `JOB_ID_TO_ATTACH`.")
            # executor = SequentialConfigExecutor(configurator_outputs=self.configurator_outputs, job_id_to_attach=job_id_to_attach)
            executor = TaskSpoolerSequentialConfigExecutor(configurator_outputs=self.configurator_outputs)
            executor.execute()
            yield rx.toast.success("Your jobs are successfully submitted!")

    @rx.event
    async def set_current_tab(self, value: str):
        self.current_tab = value

    def _get_serializable_dict(self):
        """Converts the state into a JSON-ready dictionary."""
        # Define fields to exclude (like is_loading or computed rx.vars)
        exclude = ["is_loading", "parent_state", "router_data", "substates", "dirty_vars", "dirty_substates", "router", "is_hydrated"]

        state_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            if key in exclude:
                continue

            # 1. Convert Sets to Lists
            if isinstance(value, set):
                state_dict[key] = list(value)

            # 2. Convert DataFrames
            elif isinstance(value, pd.DataFrame):
                state_dict[key] = value.to_dict(orient="records")

            # 3. Handle Custom Objects
            elif key == "configurator_outputs" and value:
                state_dict[key] = [obj.dict() if hasattr(obj, "dict") else str(obj) for obj in value]
            else:
                state_dict[key] = value

        return state_dict
