import asyncio
import dataclasses
import re
from typing import Any, Dict, List

import pandas as pd
import plotly
import plotly.graph_objects as go
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
from auto_llm.tracker.tracker import WandbClient

from ..state.app_state import AppState
from ..state.user import User, get_wandb_client, get_wandb_entity

# from ..backend.wandb_client import Client

GPU_PARAMS = get_gpu_params()


# Columns of the W&B eval-results table that are not scores
EXAMPLE_NON_SCORE_COLS = [
    "id",
    "data",
    "choices",
    "input_len",
    "labels",
    "output_type",
    "raw_predictions",
    "filtered_predictions",
    "run_name",
]

# Metrics where 1 means correct and 0 means wrong
BINARY_METRICS = ["acc", "acc_norm", "exact_match"]

# W&B run states after which a run's status no longer changes
FINAL_RUN_STATES = {"finished", "crashed", "failed", "killed"}

# Run overviews of runs in a final state never change, so they are cached per (entity, project, run_id)
_run_view_cache: Dict[tuple, Dict[str, Any]] = {}
# Creating a wandb.Api takes ~0.7s, so keep one client per W&B entity (team)
_wandb_clients: Dict[str, WandbClient] = {}


def _get_cached_client(entity: str) -> WandbClient:
    if entity not in _wandb_clients:
        _wandb_clients[entity] = WandbClient(entity=entity)
    return _wandb_clients[entity]


@dataclasses.dataclass
class ExampleScore:
    name: str
    value: str
    color: str


@dataclasses.dataclass
class ExampleResult:
    model_label: str
    run_name: str
    output: str
    scores: List[ExampleScore]


@dataclasses.dataclass
class InsightRun:
    kind: str
    kind_label: str
    model: str
    run_name: str
    value: str
    bar: int
    is_best: bool


@dataclasses.dataclass
class MetricInsight:
    metric: str
    best_value: str
    direction: str
    runs: List[InsightRun]
    verdict: str
    tone: str


@dataclasses.dataclass
class RunInfoItem:
    label: str
    value: str


@dataclasses.dataclass
class RunStat:
    section: str
    name: str
    value: str
    stderr: str


@dataclasses.dataclass
class Example:
    sample_id: str
    input_text: str
    expected_output: str
    results: List[ExampleResult]


def _model_label(run_name: str) -> str:
    if run_name.startswith("pre_"):
        return "Pre-trained"
    if run_name.startswith("fft_"):
        return "Full fine-tuned"
    if run_name.startswith("lora_"):
        return "LoRA fine-tuned"
    return run_name


def _parse_choices(choices: Any) -> Dict[str, str]:
    """Parses '0.  yes\n1.  no' into {'0': 'yes', '1': 'no'}."""
    parsed = {}
    for line in str(choices or "").splitlines():
        match = re.match(r"\s*(\d+)\.\s*(.*)", line)
        if match:
            parsed[match.group(1)] = match.group(2).strip()
    return parsed


def _to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else ", ".join(str(v) for v in value)
    return str(value).strip()


def _color_scores_relative(results: List[ExampleResult]):
    """Colors each non-binary metric by how a model compares to the others: best green, worst red, rest amber."""
    values_by_metric: Dict[str, List[float]] = {}
    for result in results:
        for score in result.scores:
            if score.name not in BINARY_METRICS:
                values_by_metric.setdefault(score.name, []).append(float(score.value))

    for result in results:
        for score in result.scores:
            values = values_by_metric.get(score.name)
            if not values or min(values) == max(values):
                continue
            value = float(score.value)
            if value == max(values):
                score.color = "green"
            elif value == min(values):
                score.color = "red"
            else:
                score.color = "amber"


def build_examples(examples_df: pd.DataFrame) -> List[Example]:
    if examples_df.empty or "id" not in examples_df.columns:
        return []

    score_cols = [c for c in examples_df.columns if c not in EXAMPLE_NON_SCORE_COLS]
    examples = []
    for sample_id, group_df in examples_df.groupby("id", sort=False):
        first = group_df.iloc[0]
        choices = _parse_choices(first.get("choices"))

        expected = _to_text(first.get("labels", ""))
        expected = choices.get(expected, expected)

        results = []
        for _, item in group_df.iterrows():
            output = _to_text(item.get("filtered_predictions", ""))
            if output in choices:
                output = f"{choices[output]} ({output})"

            scores = []
            for col in score_cols:
                value = item[col]
                if value == "" or value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                color = "gray"
                if col in BINARY_METRICS:
                    color = "green" if number >= 1 else "red"
                scores.append(ExampleScore(name=col, value=f"{number:.2f}", color=color))

            results.append(
                ExampleResult(
                    model_label=_model_label(item["run_name"]),
                    run_name=item["run_name"],
                    output=output or "(empty)",
                    scores=scores,
                )
            )

        _color_scores_relative(results)

        examples.append(
            Example(
                sample_id=str(sample_id),
                input_text=_to_text(first.get("data", "")),
                expected_output=expected,
                results=results,
            )
        )

    return examples


class ConfigurationState(rx.State):
    current_yaml_content: str = ""
    current_path: str = ""

    current_gpu_name: str = ""
    current_gpu_count: int = 0

    est_runtime: str = ""
    est_emission: str = ""

    # In-app W&B run overview (Validate tab)
    run_view_id: str = ""
    run_view_name: str = ""
    run_view_mode: str = ""
    run_view_url: str = ""
    run_view_state: str = ""
    run_view_info: List[RunInfoItem] = []
    run_view_stats: List[RunStat] = []
    run_view_fig: go.Figure = go.Figure()
    run_view_has_chart: bool = False
    run_view_error: str = ""
    is_loading_run_view: bool = False

    result_fig: str = ""
    result_insights: List[MetricInsight] = []
    examples_df: pd.DataFrame = pd.DataFrame()
    examples: List[Example] = []
    example_index: int = 0

    is_polling: bool = False
    is_loading_results: bool = False
    is_results_loaded: bool = False
    is_loading_examples_html: bool = False

    config_statuses: Dict[str, str] = {}

    # Incremented on every start/stop so that only the most recently started poll loop keeps running
    _poll_generation: int = 0

    @rx.event
    def reset_state(self):
        self.current_yaml_content: str = ""
        self.current_path: str = ""

        self.current_gpu_name: str = ""
        self.current_gpu_count: int = 0

        self.est_runtime: str = ""
        self.est_emission: str = ""

        self.result_fig: str = ""
        self.result_insights = []
        self.examples_df: pd.DataFrame = pd.DataFrame()
        self.examples = []
        self.example_index = 0

        self.is_polling: bool = False
        self.is_results_loaded: bool = False
        self.is_loading_examples_html: bool = False

        self.config_statuses: Dict[str, str] = {}

    @rx.event
    def reset_results_state(self):
        self.result_fig: str = ""
        self.result_insights = []
        self.examples_df: pd.DataFrame = pd.DataFrame()
        self.examples = []
        self.example_index = 0

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

    @rx.event
    async def load_run_view(self, run_id: str, run_name: str, mode: str):
        """Loads the in-app overview of a W&B run: info, config highlights, final values and curves."""
        self.run_view_id, self.run_view_name, self.run_view_mode = run_id, run_name, mode
        self.run_view_url = ""
        self.run_view_state = ""
        self.run_view_info = []
        self.run_view_stats = []
        self.run_view_fig = go.Figure()
        self.run_view_has_chart = False
        self.run_view_error = ""
        self.is_loading_run_view = True
        yield

        user_state = await self.get_state(User)
        entity = get_wandb_entity(user_state.active_team)
        project_name = WANDB_TRAIN_PROJECT if mode == ConfigMode.TRAINER_RUN_CFG.value else WANDB_EVAL_PROJECT
        cache_key = (entity, project_name, run_id)

        try:
            view = _run_view_cache.get(cache_key)
            if view is None:
                loop = asyncio.get_running_loop()
                client = await loop.run_in_executor(_executor, _get_cached_client, entity)
                view = await loop.run_in_executor(_executor, client.get_run_view, run_id, project_name)
                if view["state"] in FINAL_RUN_STATES:
                    _run_view_cache[cache_key] = view

            self.run_view_url = view["url"]
            self.run_view_state = view["state"]
            self.run_view_info = [RunInfoItem(**i) for i in view["info"]]
            self.run_view_stats = [RunStat(**st) for st in view["stats"]]
            if view["figure"] is not None:
                self.run_view_fig = view["figure"]
                self.run_view_has_chart = True
        except Exception as e:
            if "Could not find run" in str(e):
                self.run_view_error = "This run has not started logging to W&B yet. It is still queued or starting up."
            else:
                print(f"Error loading run view: {e}")
                self.run_view_error = f"Could not load the run from W&B: {e}"
        finally:
            self.is_loading_run_view = False

    @rx.event
    async def refresh_run_view(self):
        if self.run_view_id:
            return ConfigurationState.load_run_view(self.run_view_id, self.run_view_name, self.run_view_mode)

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
            result_fig, insights, examples_df = await loop.run_in_executor(_executor, client.get_eval_runs_of_group, group, project_name)

            if not result_fig:
                yield rx.toast.warning("No results found. Please retry later!")
                return

            self.result_fig = result_fig
            self.result_insights = [MetricInsight(**{**i, "runs": [InsightRun(**r) for r in i["runs"]]}) for i in insights]
            self.examples_df = examples_df
            self.examples = build_examples(examples_df)
            self.example_index = 0

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

    @rx.event
    def start_polling(self):
        """Starts a status poll loop for the mounted jobs table, superseding any older loop."""
        self._poll_generation += 1
        self.is_polling = True
        return ConfigurationState.poll_loop(self._poll_generation)

    @rx.event
    def stop_polling(self):
        self._poll_generation += 1
        self.is_polling = False

    @rx.event(background=True)
    async def poll_loop(self, generation: int):
        # Must be a background event: only there does `async with self` lock the state and
        # push the changes to the browser on exit. A plain asyncio task updates the state in
        # memory, but the browser only sees it when some other event happens to run.
        loop = asyncio.get_running_loop()
        client = None

        while True:
            try:
                # 1. Grab snapshot data from states quickly while inside the state lock
                async with self:
                    if not self.is_polling or self._poll_generation != generation:
                        return
                    form_state = await self.get_state(AppState)
                    user_state = await self.get_state(User)
                    entity = get_wandb_entity(user_state.active_team)

                    run_ids_by_project = {WANDB_TRAIN_PROJECT: [], WANDB_EVAL_PROJECT: []}
                    for cfg in form_state.configurator_outputs:
                        project_name = WANDB_TRAIN_PROJECT if cfg.mode == ConfigMode.TRAINER_RUN_CFG else WANDB_EVAL_PROJECT
                        run_ids_by_project[project_name].append(cfg.run_id)

                # 2. Perform blocking W&B calls outside the state lock: one query per project instead of one per run
                if client is None or client.get_entity() != entity:
                    client = await loop.run_in_executor(_executor, WandbClient, entity)

                new_statuses = {}
                for project_name, run_ids in run_ids_by_project.items():
                    states = await loop.run_in_executor(_executor, client.get_run_states, run_ids, project_name)
                    # Runs not created in W&B yet are still queued or starting up
                    new_statuses.update({run_id: states.get(run_id, "pending") for run_id in run_ids})

                # 3. Re-acquire the lock briefly to write the updates; leaving the block pushes them to the browser
                async with self:
                    if not self.is_polling or self._poll_generation != generation:
                        return
                    self.config_statuses.update(new_statuses)

                    # Nothing left to change once every run has ended
                    if new_statuses and all(state in FINAL_RUN_STATES for state in new_statuses.values()):
                        self.is_polling = False
                        return

            except Exception as e:
                print(f"Error in background poll loop: {e}")

            await asyncio.sleep(5)

    @rx.var
    def current_example(self) -> Example:
        if not self.examples:
            return Example(sample_id="", input_text="", expected_output="", results=[])
        return self.examples[min(self.example_index, len(self.examples) - 1)]

    @rx.var
    def num_examples(self) -> int:
        return len(self.examples)

    @rx.event
    def next_example(self):
        if self.example_index < len(self.examples) - 1:
            self.example_index += 1

    @rx.event
    def prev_example(self):
        if self.example_index > 0:
            self.example_index -= 1
