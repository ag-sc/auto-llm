import datetime
from typing import Optional, List, Tuple, Any

import pandas as pd
import reflex as rx
import yaml

from auto_llm.automator.automator import Automator
from auto_llm.configurator.config_generator import TrainEvalRunConfigurator, ConfiguratorOutput, ConfigMode, Priority
from auto_llm.dto.builder_config import TrainerDataBuilderConfig
from auto_llm.estimator.utils import get_gpu_params
from auto_llm.tasks.registry import TASKS
from ..components.status_badge import status_badge
from ..templates import template

GPU_PARAMS = get_gpu_params()
CONFIGS_DIR = ".cache"
OUTPUT_DIR = ".cache/sft_models/"

class FormState(rx.State):
    current_tab = "settings"

    is_loading: bool = False
    model_choices: list[str] = []
    model_results: pd.DataFrame = pd.DataFrame()

    # settings tab
    dataset_path: Optional[str] = ""
    task_category: Optional[str] = ""
    hardware_type: Optional[str] = ""
    hardware_count: Optional[str] = ""

    # models tab
    selected_model: Optional[str] = ""

    # prompts tab
    instruction_template: Optional[str] = ""
    input_template: Optional[str] = ""
    output_template: Optional[str] = ""

    configurator_outputs: Optional[List[ConfiguratorOutput]] = []
    configs_path: Optional[str] = ""

    @rx.event
    def reset_state(self):
        self.current_tab = "settings"

        self.is_loading: bool = False
        self.model_choices: list[str] = []
        self.model_results: pd.DataFrame = pd.DataFrame()

        # settings tab
        self.dataset_path: Optional[str] = ""
        self.task_category: Optional[str] = ""
        self.hardware_type: Optional[str] = ""
        self.hardware_count: Optional[str] = ""

        # models tab
        self.selected_model: Optional[str] = ""

        # prompts tab
        self.instruction_template: Optional[str] = ""
        self.input_template: Optional[str] = ""
        self.output_template: Optional[str] = ""

        self.configurator_outputs: Optional[List[ConfiguratorOutput]] = []
        self.configs_path: Optional[str] = ""


    @rx.var
    def dataset_options_markdown(self) -> str:
        datasets = Automator.get_datasets()
        prefix = "https://huggingface.co/datasets"
        return "\n".join([f"* [`{d}`]({prefix}/{d})" for d in datasets])

    @rx.event
    async def handle_submit(self, form_data: dict):
        if not all([form_data.get("dataset_path"), form_data.get("task_category"), form_data.get("hardware_type")]):
            yield rx.window_alert("Please fill in all required fields!")

        self.is_loading = True
        yield

        # Logic to fetch models
        model_names, results_df = update_models(
            task=self.task_category,
            dataset=self.dataset_path,
            hardware_type=self.hardware_type,
            hardware_count=int(self.hardware_count)
        )

        self.model_choices = model_names
        self.model_results = results_df
        self.is_loading = False
        self.current_tab = "models"

        configured_task = TASKS.get(self.task_category)

        self.instruction_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.instruction_template
        self.input_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.input_template
        self.output_template = configured_task.sample_trainer_run_config.trainer_data_builder_config.output_template

    @rx.event
    async def handle_models_submit(self, form_data: dict):
        self.current_tab = "prompts"

    @rx.event
    async def handle_prompts_submit(self, form_data: dict):
        configurator_outputs = generate_configs(
            model_names=[self.selected_model],
            task=self.task_category,
            dataset_path=self.dataset_path,
            instruction_template=self.instruction_template,
            input_template=self.input_template,
            output_template=self.output_template,
        )

        sorted_configurator_outputs = []
        for p in [Priority.PRIORITY_ONE, Priority.PRIORITY_TWO, Priority.PRIORITY_THREE]:
            for co in configurator_outputs:
                if co.priority == p:
                    sorted_configurator_outputs.append(co)


        self.configurator_outputs = sorted_configurator_outputs


        self.current_tab = "validate"

    @rx.event
    async def handle_validation_submit(self, form_data: dict):
        self.current_tab = "execute"

    @rx.event
    async def set_current_tab(self, value: str):
        self.current_tab = value


def update_models(task: str, dataset: str, hardware_type: str, hardware_count: int):
    # This remains similar to your logic but ensures clean data for the table
    configured_task = TASKS.get(task)
    automator = Automator(
        task_type=configured_task.name,
        dataset=dataset,
        hardware_type=hardware_type,
        hardware_count=hardware_count,
    )
    df = automator.get_models_df()
    # Clean up DF columns for display
    cols = [df.columns[0]] + list(df.columns[2:])

    df = df.round(2)
    return automator.model_names, df[cols]


def generate_configs(
    model_names: List[str],
    task: str,
    dataset_path: str,
    instruction_template: str,
    input_template: str,
    output_template: str,
) -> List[ConfiguratorOutput]:
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    configs_path = f"{CONFIGS_DIR}/{timestamp_str}_configs"

    configurator = TrainEvalRunConfigurator(
        model_names=model_names,
        task=task,
        dataset_path=dataset_path,
        configs_path=configs_path,
        output_path=OUTPUT_DIR,
        instruction_template=instruction_template,
        input_template=input_template,
        output_template=output_template,
    )

    configurator_outputs = configurator.generate()
    return configurator_outputs




def info_popover(title: str, content: str):
    """Refactored helper for clean info popovers."""
    return rx.popover.root(
        rx.popover.trigger(
            rx.icon("info", size=16, color_scheme="gray", cursor="pointer")
        ),
        rx.popover.content(
            rx.vstack(
                rx.heading(title, size="3"),
                rx.markdown(content),
                spacing="2",
            ),
            width="300px",
        ),
    )


def _header_cell(text: str, icon: str):
    return rx.table.column_header_cell(
        rx.hstack(
            rx.icon(icon, size=18),
            rx.text(text),
            align="center",
            spacing="2",
        ),
    )

def view_yaml_file(path: str):
    print("path", path)
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data = yaml.dump(data)
    except FileNotFoundError:
        print("File not found", path)
        data = ""
    return data

def dialog_popover(config_path: str, config_yaml: str):
    return rx.dialog.root(
        rx.dialog.trigger(rx.button(rx.icon("view"), variant="soft", size="1")),
        rx.dialog.content(
            rx.heading("Configuration Viewer"),
            rx.markdown(f"Path: **{config_path}**"),
            rx.code_block(config_yaml),
            rx.dialog.close(rx.button("Close", mt="4")),
            size="4"
        ),
        width="300px",
    ),

def show_configuration(configurator_output: ConfiguratorOutput):
    """Show a customer in a table row."""

    config_yaml = view_yaml_file(path=rx.text(configurator_output.config_path))

    return rx.table.row(
        rx.table.cell(
            rx.hstack(
                rx.text(configurator_output.run_name),
                dialog_popover(config_path=configurator_output.config_path, config_yaml=config_yaml),
            )

        ),
        rx.table.cell(
            rx.match(
                configurator_output.mode,
                (ConfigMode.TRAINER_RUN_CFG, status_badge(ConfigMode.TRAINER_RUN_CFG.name)),
                (ConfigMode.EVALUATOR_RUN_CFG, status_badge(ConfigMode.EVALUATOR_RUN_CFG.name)),
            )
        ),
        rx.table.cell(
            rx.match(
                configurator_output.priority,
                (Priority.PRIORITY_ONE, status_badge("1")),
                (Priority.PRIORITY_TWO, status_badge("2")),
                (Priority.PRIORITY_THREE, status_badge("3")),
            )
        ),
        rx.table.cell(
            rx.match(
                configurator_output.priority,
                ("1", status_badge("Delivered")),
                ("2", status_badge("Pending")),
                ("3", status_badge("Cancelled")),
                status_badge("Pending"),
            )
        ),
        # rx.table.cell(
        #     rx.hstack(
        #         update_customer_dialog(user),
        #         rx.icon_button(
        #             rx.icon("trash-2", size=22),
        #             on_click=lambda: State.delete_customer(user.id),
        #             size="2",
        #             variant="solid",
        #             color_scheme="red",
        #         ),
        #     )
        # ),
        style={"_hover": {"bg": rx.color("gray", 3)}},
        align="center",
    )

@template(route="/configure", title="Configure", on_load=FormState.reset_state)
def configure() -> rx.Component:
    dataset_path_descr = f"""
    Path of the dataset. This can either be remote HuggingFace Datasets paths or local paths. 

    Examples:
    {FormState.dataset_options_markdown}
    """


    tasks_descr = """
    Category of the task. This can either be:
    - **Sequence To Sequence**
    - **Sequence To Label**
    - **Sequence To Structured Output**
    """

    form = rx.form(
                rx.card(
                    rx.vstack(
                        # --- Section: Dataset ---
                        rx.vstack(
                            rx.hstack(rx.icon("database", size=20), rx.text("Dataset Path", weight="bold"), info_popover("Datasets", dataset_path_descr), align="center",),
                            rx.input(placeholder="e.g. llm-4-kmu/pubmed_mcqa", name="dataset_path", value=FormState.dataset_path, on_change=FormState.setvar("dataset_path"), width="100%", variant="surface", size="3",),
                            width="100%", align_items="start",
                        ),
                        rx.vstack(
                            rx.hstack(rx.icon("layers", size=20), rx.text("Task Category", weight="bold"), info_popover("Task Categories", tasks_descr), align="center",),
                            rx.select(list(TASKS.keys()), placeholder="Select task...", name="task_category", value=FormState.task_category, on_change=FormState.setvar("task_category"), width="100%", variant="surface", size="3",),
                            width="100%", align_items="start",
                        ),

                        # --- Section: Task & Hardware ---
                        rx.grid(
                            rx.vstack(
                                rx.hstack(rx.icon("microchip", size=20), rx.text("Hardware Type", weight="bold")),
                                rx.select(GPU_PARAMS, placeholder="Select GPU...", name="hardware_type", value=FormState.hardware_type, on_change=FormState.setvar("hardware_type"), width="100%", size="3",),
                                align_items="start",
                            ),
                            rx.vstack(
                                rx.hstack(rx.icon("hash", size=20), rx.text("Hardware Count", weight="bold")),
                                rx.select(["1", "2", "4", "8"], placeholder="1", name="hardware_count", value=FormState.hardware_count, on_change=FormState.setvar("hardware_count"), width="100%", size="3",),
                                align_items="start",
                            ),
                            columns="2", spacing="4", width="100%",
                        ),

                        rx.button(
                            "Next",
                            type="submit",
                            width="100%",
                            size="3",
                            variant="solid",
                        ),
                        spacing="5",
                        padding="4",
                    ),
                    width="60vw",
                ),
                on_submit=FormState.handle_submit,
            )

    model_results = rx.form(
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("cpu", size=20), rx.text("Recommended Models", weight="bold"),
                            rx.spacer(),
                            rx.dialog.root(
                                rx.dialog.trigger(rx.button("View Benchmarks", variant="soft", size="1")),
                                rx.dialog.content(
                                    rx.heading("Model Results"),
                                    rx.markdown(
                                        "We have pre-selected the following models based on the task category and hardware parameters you entered. "
                                        "We relied the selection based on the results from [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)."),
                                    rx.data_table(data=FormState.model_results, resizable=True, pagination=True),
                                    rx.dialog.close(rx.button("Close", mt="4")),
                                    size="4"
                                ),
                            ),
                            width="100%",
                        ),
                        rx.select(
                            FormState.model_choices,
                            placeholder="Select a suggested model",
                            width="100%",
                            size="3",
                            value=FormState.selected_model,
                            on_change=FormState.setvar("selected_model"),
                        ),
                        rx.button(
                            "Next",
                            type="submit",
                            width="100%",
                            size="3",
                            variant="solid",
                        ),
                        width="100%",
                    ),
                    width="60vw",
                    margin_top="4",
                ),

        spacing="5",
        padding="4",
        width = "60vw",
        on_submit = FormState.handle_models_submit,

    )


    prompt_templates = rx.form(
        rx.card(
            rx.vstack(
                rx.hstack(rx.icon("layout-template", size=20), rx.text("Instruction Template", weight="bold"),
                          rx.spacer(),
                          rx.dialog.root(
                              rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                              rx.dialog.content(
                                  rx.dialog.title("Guidelines"),
                                  rx.markdown(TrainerDataBuilderConfig.model_fields["instruction_template"].description,),
                                  rx.dialog.close(rx.button("Close", mt="4")),
                                  size="4"
                              ),
                          ),
                          width="100%",
                          ),
                rx.text_area(
                    placeholder="",
                    name="instruction_template",
                    value=FormState.instruction_template,
                    on_change=FormState.setvar("instruction_template"),
                    size="3", width="100%", rows="5"
                ),

                rx.hstack(rx.icon("layout-template", size=20), rx.text("Input Template", weight="bold"),
                          rx.spacer(),
                          rx.dialog.root(
                              rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                              rx.dialog.content(
                                  rx.dialog.title("Guidelines"),
                                  rx.markdown(
                                      TrainerDataBuilderConfig.model_fields["input_template"].description, ),
                                  rx.dialog.close(rx.button("Close", mt="4")),
                                  size="4"
                              ),
                          ),
                          width="100%",
                          ),
                rx.text_area(
                    placeholder="",
                    name="input_template",
                    value=FormState.input_template,
                    on_change=FormState.setvar("input_template"),
                    size="3", width="100%", rows="5"
                ),

                rx.hstack(rx.icon("layout-template", size=20), rx.text("Output Template", weight="bold"),
                          rx.spacer(),
                          rx.dialog.root(
                              rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                              rx.dialog.content(
                                  rx.dialog.title("Guidelines"),
                                  rx.markdown(
                                      TrainerDataBuilderConfig.model_fields["output_template"].description, ),
                                  rx.dialog.close(rx.button("Close", mt="4")),
                                  size="4"
                              ),
                          ),
                          width="100%",
                          ),
                rx.text_area(
                    placeholder="",
                    name="output_template",
                    value=FormState.output_template,
                    on_change=FormState.setvar("output_template"),
                    size="3", width="100%", rows="5"
                ),

                rx.button(
                    "Next",
                    type="submit",
                    width="100%",
                    size="3",
                    variant="solid",
                ),
                width="100%", align_items="column",
            ),
            width="60vw",
            margin_top="4",

        ),
        spacing="5",
        padding="4",
        width="60vw",
        on_submit=FormState.handle_prompts_submit,
    )


    jobs_table =  rx.table.root(
            rx.table.header(
                rx.table.row(
                    _header_cell("Run Name", "fingerprint"),
                    _header_cell("Mode", "beaker"),
                    _header_cell("Priority", "gauge"),
                    _header_cell("Status", "cog"),
                ),
            ),
            rx.table.body(rx.foreach(FormState.configurator_outputs, show_configuration)),
            variant="surface",
            size="3",
            width="100%",
        ),

    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Settings", value="settings"),
            rx.tabs.trigger("Models", value="models"),
            rx.tabs.trigger("Prompts", value="prompts"),
            rx.tabs.trigger("Validate", value="validate"),
            # rx.tabs.trigger("Execute", value="execute"),
            ),

            rx.tabs.content(
                form,
                value="settings",
            ),
            rx.tabs.content(
                model_results,
                value="models",
            ),
            rx.tabs.content(
                prompt_templates,
                value="prompts",
            ),
            rx.tabs.content(
                jobs_table,
                value="validate",
            ),

            default_value="settings",
            value=FormState.current_tab,
            on_change=FormState.set_current_tab
        )