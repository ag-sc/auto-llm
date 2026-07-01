import asyncio
from typing import Any, Dict

import pandas as pd
import plotly
import reflex as rx
import yaml

import asyncio
from concurrent.futures import ThreadPoolExecutor

# A shared executor to handle blocking W&B API network calls safely
_executor = ThreadPoolExecutor(max_workers=5)

from auto_llm.configurator.config_generator import ConfigMode, ConfiguratorOutput
from auto_llm.estimator.emission_estimator import EmissionEstimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator
from auto_llm.estimator.utils import get_gpu_params, get_model_params
from auto_llm.registry.tracker_registry import WANDB_TRAIN_PROJECT, WANDB_EVAL_PROJECT

from ..state.app_state import AppState
from ..state.user import User, get_wandb_client

# from ..backend.wandb_client import Client

GPU_PARAMS = get_gpu_params()


TEMPLATE = """\
## Example {idx}

### **Input Text**
{input_text}

### **Expected Output:**
{expected_output}

---

### **Results**
{results}
"""

RESULTS_TEMPLATE = """\
**Model:** ``{run_name}``

**Output:**<br>
{generated_output}

**Scores:**
```json
{scores}
```

---
"""


class ConfigurationState(rx.State):
    current_yaml_content: str = ""
    current_path: str = ""

    current_gpu_name: str = ""
    current_gpu_count: int = 0

    est_runtime: str = ""
    est_emission: str = ""

    current_html_content: str = ""
    result_fig: str = ""
    result_explanation: str = ""
    examples_df: pd.DataFrame = pd.DataFrame()
    examples_html: str = ""

    is_polling: bool = False
    is_loading_results: bool = False
    is_results_loaded: bool = False
    is_loading_examples_html: bool = False

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
        self.result_explanation: str = ""
        self.examples_df: pd.DataFrame = pd.DataFrame()
        self.examples_html: str = ""

        self.is_polling: bool = False
        self.is_results_loaded: bool = False
        self.is_loading_examples_html: bool = False

        self.config_statuses: Dict[str, str] = {}

    @rx.event
    def reset_results_state(self):
        self.result_fig: str = ""
        self.result_explanation: str = ""
        self.examples_df: pd.DataFrame = pd.DataFrame()
        self.examples_html: str = ""

        self.is_polling: bool = False
        self.is_results_loaded: bool = False
        self.is_loading_examples_html: bool = False

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

    async def load_config_statuses(self):
        form_state = await self.get_state(AppState)
        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)

        for cfg in form_state.configurator_outputs:
            try:
                project_name = WANDB_TRAIN_PROJECT if cfg.mode == ConfigMode.TRAINER_RUN_CFG else WANDB_EVAL_PROJECT

                # Run the blocking synchronous W&B call in a separate thread safely
                state = await asyncio.to_thread(client.get_run_state, run_id=cfg.run_id, project_name=project_name)

                self.config_statuses[cfg.run_id] = state
            except Exception:
                self.config_statuses[cfg.run_id] = "pending"

    async def load_config_html(self, configurator_output: ConfiguratorOutput):
        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)
        project_name = WANDB_TRAIN_PROJECT if configurator_output.mode == ConfigMode.TRAINER_RUN_CFG else WANDB_EVAL_PROJECT

        # Run the blocking synchronous W&B call in a separate thread safely
        run_html = await asyncio.to_thread(
            client.get_run_url,
            run_id=configurator_output.run_id,
            project_name=project_name,
        )

        self.current_html_content = f'<iframe src="{run_html}" ' f'style="width:100%; height:80vh; border:none; display:block;" ' f"allowfullscreen></iframe>"

    @rx.event
    async def load_config_group_results(self, group: str):
        self.is_loading_results = True
        self.is_results_loaded = False
        self.result_fig = ""
        yield

        loop = asyncio.get_event_loop()
        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)
        project_name = WANDB_EVAL_PROJECT

        try:
            # Use the explicit thread executor instead of None
            result_fig, result_explanation, examples_df = await loop.run_in_executor(_executor, client.get_eval_runs_of_group, group, project_name)

            self.result_fig = result_fig
            self.result_explanation = result_explanation
            self.examples_df = examples_df
            self.examples_html = self.display_examples()

        except Exception as e:
            print(f"Error loading W&B: {e}")
            if self.result_fig == "":
                yield rx.toast.warning("No results found. Please retry later!")
        finally:
            self.is_loading_results = False
            self.is_results_loaded = True

            # print("is_results_loaded", self.is_results_loaded)
            # print("is_loading_results", self.is_loading_results)
            yield  # CRITICAL: Forces UI to register that loading has finished

    async def start_polling(self):
        """This starts the loop if it's not already running."""
        if self.is_polling:
            return
        self.is_polling = True
        asyncio.create_task(self.poll_loop())

    async def poll_loop(self):
        loop = asyncio.get_event_loop()

        while self.is_polling:
            try:
                # 1. Grab snapshot data from states quickly while inside the context lock
                async with self:
                    form_state = await self.get_state(AppState)
                    user_state = await self.get_state(User)
                    client = get_wandb_client(user=user_state)

                    # Create a list of configurations to check so we can release the lock
                    configs_to_check = [
                        (cfg.run_id, WANDB_TRAIN_PROJECT if cfg.mode == ConfigMode.TRAINER_RUN_CFG else WANDB_EVAL_PROJECT) for cfg in form_state.configurator_outputs
                    ]

                # 2. Perform blocking I/O network calls OUTSIDE the state lock
                new_statuses = {}
                for run_id, project_name in configs_to_check:
                    try:
                        # Offload blocking network call to thread pool
                        state = await loop.run_in_executor(_executor, client.get_run_state, run_id, project_name)
                        new_statuses[run_id] = state
                    except Exception:
                        new_statuses[run_id] = "pending"

                # 3. Re-acquire the lock briefly just to write the updates to state variables
                async with self:
                    if not self.is_polling:  # Guard check in case polling stopped during I/O
                        break
                    self.config_statuses.update(new_statuses)

            except Exception as e:
                print(f"Error in background poll loop: {e}")

            await asyncio.sleep(5)

    def display_examples(self):
        samples_text = []

        cols_to_drop = [
            "id",
            "data",
            "input_len",
            "labels",
            "output_type",
            "raw_predictions",
            "filtered_predictions",
        ]
        for idx, group_df in self.examples_df.groupby("id"):
            result_texts = []
            for idx, item in group_df.iterrows():
                filtered_item = item.copy()
                for col in cols_to_drop:
                    filtered_item.pop(col)

                scores_json = filtered_item.to_json(indent=4)
                result = dict(
                    run_name=item["run_name"],
                    generated_output=item["filtered_predictions"],
                    scores=scores_json,
                )
                result_text = RESULTS_TEMPLATE.format(**result)

                result_texts.append(result_text)

            sample = dict(
                idx=idx,
                input_text=group_df.iloc[0]["data"],
                expected_output=group_df.iloc[0]["labels"],
                results="\n".join(result_texts),
            )

            sample_text = TEMPLATE.format(**sample)
            samples_text.append(sample_text)

        return samples_text[0]
