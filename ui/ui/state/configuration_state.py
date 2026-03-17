import asyncio
from typing import Any, Dict

import plotly
import reflex as rx
import yaml

from auto_llm.configurator.config_generator import ConfiguratorOutput
from auto_llm.estimator.emission_estimator import EmissionEstimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator
from auto_llm.estimator.utils import get_gpu_params, get_model_params
from auto_llm.registry.tracker_registry import WANDB_PROJECT

from ..state.app_state import AppState
from ..backend.wandb_client import Client

GPU_PARAMS = get_gpu_params()


class ConfigurationState(rx.State):
    current_yaml_content: str = ""
    current_path: str = ""

    current_gpu_name: str = ""
    current_gpu_count: int = 0

    est_runtime: str = ""
    est_emission: str = ""

    current_html_content: str = ""
    result_fig: str = ""

    is_polling: bool = False

    config_statuses: Dict[str, str] = {}

    @rx.event
    def reset_state(self):
        self.current_yaml_content: str = ""
        self.current_path: str = ""

        self.current_gpu_name: str = ""
        self.current_gpu_count: int = 0

        self.est_runtime: str = ""
        self.est_emission: str = ""

        self.current_html_content: str = ""
        self.result_fig: str = ""

        self.is_polling: bool = False

        self.config_statuses: Dict[str, str] = {}

    def load_config(self, configurator_output: ConfiguratorOutput):
        self.current_path = configurator_output.config_path
        with open(configurator_output.config_path, "r") as f:
            data = yaml.safe_load(f)
            self.current_yaml_content = yaml.dump(data)

    def update_content(self, new_value: str):
        """Update the state as the user types."""
        self.current_yaml_content = new_value

    def save_config(self):
        """Save the edited changes back to the file."""
        try:
            with open(self.current_path, "w") as f:
                f.write(self.current_yaml_content)
            return rx.toast("File saved successfully!")
        except Exception as e:
            return rx.toast(f"Error saving: {e}")

    def load_estimates(self, path: str, gpu_name: str, gpu_count: int):
        self.current_path = path
        self.current_gpu_name = gpu_name
        self.current_gpu_count = gpu_count

        models_meta = get_model_params()

        if "eval" in self.current_path:
            flops_estimator = InferenceFlopsEstimator(config_path=self.current_path, models_meta=models_meta)
        elif "train" in self.current_path:
            # TODO: models_meta is not updated with the requested model
            flops_estimator = TrainerFlopsEstimator(config_path=self.current_path, models_meta=models_meta)
        else:
            self.est_runtime = f"-1 seconds"
            self.est_emission = f"-1 grams"

            return rx.toast(f"Error estimating.")

        gpu_params = get_gpu_params()
        runtime_estimator = RuntimeEstimator(
            flops_estimator=flops_estimator,
            gpu_params=gpu_params,
            gpu_name=gpu_name,
        )
        runtime = runtime_estimator.estimate()

        emission_estimator = EmissionEstimator(
            runtime_estimator=runtime_estimator,
            gpu_params=gpu_params,
            gpu_name=gpu_name,
        )

        emission = emission_estimator.estimate()

        self.est_runtime = f"{round(runtime, 2)} seconds"
        self.est_emission = f"{round(emission, 2)} grams"

        return None

    def load_config_html(self, configurator_output: ConfiguratorOutput):
        # if configurator_output.run_id:
        #     self.current_html_content = Client.get_loss_plot(run_id=configurator_output.run_id, project_name="llm4kmu-train")
        # else:
        #     run = Client.get_run_details(run_name=configurator_output.run_name, project_name="llm4kmu-train", dt_object=datetime.datetime.now(), user_name="viju-sudhi")
        #     self.current_html_content = Client.get_run_plot_html(run)

        run_html = Client.get_run_url(
            run_id=configurator_output.run_id,
            project_name=WANDB_PROJECT,
        )
        self.current_html_content = f'<iframe src="{run_html}" ' f'style="width:100%; height:80vh; border:none; display:block;" ' f"allowfullscreen></iframe>"

    def load_config_group_results(self, group: str):
        result_fig = Client.get_eval_runs_of_group(group=group, project_name=WANDB_PROJECT)
        self.result_fig = result_fig

    async def start_polling(self):
        """This starts the loop if it's not already running."""
        if self.is_polling:
            return
        self.is_polling = True

        # We manually create a background task
        asyncio.create_task(self.poll_loop())

    async def poll_loop(self):
        while self.is_polling:
            # Sync with the State to update variables safely
            async with self:
                form_state = await self.get_state(AppState)
                for cfg in form_state.configurator_outputs:
                    try:
                        state = Client.get_run_state(run_id=cfg.run_id, project_name=WANDB_PROJECT)
                        self.config_statuses[cfg.run_id] = state
                    except Exception:
                        self.config_statuses[cfg.run_id] = "pending"

            await asyncio.sleep(5)
