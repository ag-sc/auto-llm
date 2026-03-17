import reflex as rx

from auto_llm.configurator.config_generator import ConfiguratorOutput, ConfigMode, Priority
from auto_llm.dto.builder_config import TrainerDataBuilderConfig
from auto_llm.estimator.utils import get_gpu_params
from auto_llm.tasks.registry import TASKS

from ..state.app_state import AppState
from ..state.configuration_state import ConfigurationState

from ..components.status_badge import status_badge
from ..templates import template

GPU_PARAMS = get_gpu_params()


def info_popover(title: str, content: str):
    return rx.popover.root(
        rx.popover.trigger(rx.icon("info", size=16, color_scheme="gray", cursor="pointer")),
        rx.popover.content(
            rx.vstack(
                rx.heading(title, size="3"),
                rx.markdown(content),
                spacing="2",
            ),
            width="300px",
        ),
    )


def config_html_dialog(configurator_output: ConfiguratorOutput):
    return rx.dialog.root(
        rx.dialog.trigger(rx.icon_button("view", variant="soft", size="1"), on_click=ConfigurationState.load_config_html(configurator_output)),
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title(f"View Run"),
                rx.dialog.description(f"Run: {configurator_output.run_name}"),
                rx.card(
                    rx.el.iframe(
                        src_doc=ConfigurationState.current_html_content,
                        width="100%",
                        height="100%",
                    ),
                    width="100%",
                    height="100%",
                ),
                rx.hstack(
                    rx.dialog.close(rx.button("Close", variant="soft")),
                    # rx.button("Save Changes", on_click=ConfigState.save_config),
                    justify="end",
                    width="100%",
                ),
                spacing="3",
                width="100%",
                height="100%",
            ),
            width="75vw",
            height="80vh",
            max_width="75vw",
            max_height="80vh",
            padding="2em",
        ),
    )


def config_view_dialog(configurator_output: ConfiguratorOutput):
    return rx.dialog.root(
        rx.dialog.trigger(rx.button(rx.text(configurator_output.run_name), variant="outline", size="1", on_click=ConfigurationState.load_config(configurator_output))),
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title(f"View Configuration"),
                rx.dialog.description(f"Path: {configurator_output.config_path}"),
                rx.text_area(
                    value=ConfigurationState.current_yaml_content,
                    on_change=ConfigurationState.update_content,
                    # 1. Font & Alignment
                    font_family="Source Code Pro, Menlo, Monaco, Lucide Console, monospace",
                    font_size="13px",
                    line_height="1.5",
                    # 2. Sizing & Scrolling
                    width="100%",
                    height="500px",
                    # 3. YAML Formatting Essentials (Custom CSS)
                    style={
                        "white-space": "pre",  # Crucial: Preserves leading spaces/tabs
                        "overflow_x": "auto",  # Horizontal scroll for long lines
                        "tab_size": "2",  # YAML standard is 2 spaces
                        "resize": "vertical",  # Let users pull the box larger
                        "padding": "1rem",
                        "border": "1px solid var(--gray-5)",
                        "background": "var(--gray-2)",  # Suble gray background for code
                    },
                ),
                rx.hstack(
                    rx.dialog.close(rx.button("Close", variant="soft")),
                    rx.button("Save Changes", on_click=ConfigurationState.save_config),
                    justify="end",
                    width="100%",
                ),
                spacing="3",
                padding="1rem",
            ),
            width="75vw",
            height="80vh",
            max_width="75vw",
            max_height="80vh",
            padding="2em",
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


def dialog_popover(config_path: str, config_yaml: str):
    return (
        rx.dialog.root(
            rx.dialog.trigger(rx.button(rx.icon("view"), variant="soft", size="1")),
            rx.dialog.content(
                rx.heading("Configuration Viewer"), rx.markdown(f"Path: **{config_path}**"), rx.code_block(config_yaml), rx.dialog.close(rx.button("Close", mt="4")), size="4"
            ),
            width="300px",
        ),
    )


def execute_configs_dialog():
    return (
        rx.dialog.root(
            rx.dialog.trigger(rx.button("Execute", variant="soft", size="3")),
            rx.dialog.content(
                rx.vstack(
                    rx.heading("Execution"),
                    rx.text(f"Are you sure you want to execute these configurations? This will start the runs on your specified hardware and may incur costs."),
                    rx.hstack(
                        rx.dialog.close(rx.button("Yes, Continue.", variant="soft", size="3", on_click=AppState.handle_validation_submit)),
                        rx.dialog.close(rx.button("Close", size="3")),
                        justify="center",
                    ),
                    size="4",
                    align="center",
                )
            ),
            width="300px",
        ),
    )


def show_configuration(configurator_output: ConfiguratorOutput):
    current_status = ConfigurationState.config_statuses[configurator_output.run_id]
    return rx.table.row(
        rx.table.cell(
            rx.match(
                configurator_output.priority,
                (Priority.PRIORITY_ONE.name, status_badge("1")),
                (Priority.PRIORITY_TWO.name, status_badge("2")),
                (Priority.PRIORITY_THREE.name, status_badge("3")),
            ),
        ),
        rx.table.cell(
            rx.match(
                configurator_output.mode,
                (ConfigMode.TRAINER_RUN_CFG, status_badge(ConfigMode.TRAINER_RUN_CFG.name)),
                (ConfigMode.EVALUATOR_RUN_CFG, status_badge(ConfigMode.EVALUATOR_RUN_CFG.name)),
            ),
            align="center",
        ),
        rx.table.cell(config_view_dialog(configurator_output)),
        # rx.table.cell(
        #     rx.hstack(
        #         rx.button(
        #             rx.icon("view"),
        #             variant="soft",
        #             size="1",
        #             on_click=ConfigurationState.load_estimates(path=configurator_output.config_path, gpu_name=AppState.hardware_type, gpu_count=AppState.hardware_count),
        #         ),
        #         rx.text(ConfigurationState.est_runtime),
        #     ),
        #     align="center",
        # ),
        # rx.table.cell(rx.text(ConfigurationState.est_emission)),
        rx.table.cell(
            rx.hstack(
                rx.match(
                    current_status,
                    ("running", status_badge("running")),
                    ("finished", status_badge("finished")),
                    ("failed", status_badge("failed")),
                    ("crashed", status_badge("crashed")),
                    ("killed", status_badge("killed")),
                    ("pending", status_badge("pending")),
                ),
                config_html_dialog(configurator_output),
            ),
            align="center",
        ),
        on_mount=ConfigurationState.start_polling,
        on_focus=ConfigurationState.start_polling,
        on_blur=ConfigurationState.start_polling,
        on_mouse_enter=ConfigurationState.start_polling,
        on_mouse_over=ConfigurationState.start_polling,
        on_mouse_leave=ConfigurationState.start_polling,
        on_click=ConfigurationState.start_polling,
        style={"_hover": {"bg": rx.color("gray", 3)}},
        align="center",
    )


@template(route="/configure", title="Configure", on_load=AppState.reset_state)
def configure() -> rx.Component:
    dataset_path_descr = f"""
    Path of the dataset. This can either be remote HuggingFace Datasets paths or local paths. 

    Examples:
    {AppState.dataset_options_markdown}
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
                    rx.hstack(
                        rx.icon("database", size=20),
                        rx.text("Dataset Path", weight="bold"),
                        info_popover("Datasets", dataset_path_descr),
                        align="center",
                    ),
                    rx.input(
                        placeholder="e.g. llm-4-kmu/pubmed_mcqa",
                        name="dataset_path",
                        value=AppState.dataset_path,
                        on_change=AppState.setvar("dataset_path"),
                        width="100%",
                        variant="surface",
                        size="3",
                    ),
                    width="100%",
                    align_items="start",
                ),
                rx.vstack(
                    rx.hstack(
                        rx.icon("layers", size=20),
                        rx.text("Task Category", weight="bold"),
                        info_popover("Task Categories", tasks_descr),
                        align="center",
                    ),
                    rx.select(
                        list(TASKS.keys()),
                        placeholder="Select task...",
                        name="task_category",
                        value=AppState.task_category,
                        on_change=AppState.setvar("task_category"),
                        width="100%",
                        variant="surface",
                        size="3",
                    ),
                    width="100%",
                    align_items="start",
                ),
                # --- Section: Task & Hardware ---
                rx.grid(
                    rx.vstack(
                        rx.hstack(rx.icon("microchip", size=20), rx.text("Hardware Type", weight="bold")),
                        rx.select(
                            GPU_PARAMS,
                            placeholder="Select GPU...",
                            name="hardware_type",
                            value=AppState.hardware_type,
                            on_change=AppState.setvar("hardware_type"),
                            width="100%",
                            size="3",
                        ),
                        align_items="start",
                    ),
                    rx.vstack(
                        rx.hstack(rx.icon("hash", size=20), rx.text("Hardware Count", weight="bold")),
                        rx.select(
                            ["1", "2", "4", "8"],
                            placeholder="1",
                            name="hardware_count",
                            value=AppState.hardware_count,
                            on_change=AppState.setvar("hardware_count"),
                            width="100%",
                            size="3",
                        ),
                        align_items="start",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
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
        on_submit=AppState.handle_submit,
    )

    model_results = rx.form(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("cpu", size=20),
                    rx.text("Recommended Models", weight="bold"),
                    rx.spacer(),
                    rx.dialog.root(
                        rx.dialog.trigger(rx.button("View Benchmarks", variant="soft", size="1")),
                        rx.dialog.content(
                            rx.heading("Model Results"),
                            rx.markdown(
                                "We have pre-selected the following models based on the task category and hardware parameters you entered. "
                                "We relied the selection based on the results from [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)."
                            ),
                            rx.data_table(data=AppState.model_results, resizable=True, pagination=True),
                            rx.dialog.close(rx.button("Close", mt="4")),
                            size="4",
                        ),
                    ),
                    width="100%",
                ),
                rx.select(
                    AppState.model_choices,
                    placeholder="Select a suggested model",
                    width="100%",
                    size="3",
                    value=AppState.selected_model,
                    on_change=AppState.setvar("selected_model"),
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
        width="60vw",
        on_submit=AppState.handle_models_submit,
    )

    prompt_templates = rx.form(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("layout-template", size=20),
                    rx.text("Instruction Template", weight="bold"),
                    rx.spacer(),
                    rx.dialog.root(
                        rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                        rx.dialog.content(
                            rx.dialog.title("Guidelines"),
                            rx.markdown(
                                TrainerDataBuilderConfig.model_fields["instruction_template"].description,
                            ),
                            rx.dialog.close(rx.button("Close", mt="4")),
                            size="4",
                        ),
                    ),
                    width="100%",
                ),
                rx.text_area(
                    placeholder="",
                    name="instruction_template",
                    value=AppState.instruction_template,
                    on_change=AppState.setvar("instruction_template"),
                    size="3",
                    width="100%",
                    rows="5",
                ),
                rx.hstack(
                    rx.icon("layout-template", size=20),
                    rx.text("Input Template", weight="bold"),
                    rx.spacer(),
                    rx.dialog.root(
                        rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                        rx.dialog.content(
                            rx.dialog.title("Guidelines"),
                            rx.markdown(
                                TrainerDataBuilderConfig.model_fields["input_template"].description,
                            ),
                            rx.dialog.close(rx.button("Close", mt="4")),
                            size="4",
                        ),
                    ),
                    width="100%",
                ),
                rx.text_area(
                    placeholder="",
                    name="input_template",
                    value=AppState.input_template,
                    on_change=AppState.setvar("input_template"),
                    size="3",
                    width="100%",
                    rows="5",
                ),
                rx.hstack(
                    rx.icon("layout-template", size=20),
                    rx.text("Output Template", weight="bold"),
                    rx.spacer(),
                    rx.dialog.root(
                        rx.dialog.trigger(rx.button("Guidelines", variant="soft", size="1")),
                        rx.dialog.content(
                            rx.dialog.title("Guidelines"),
                            rx.markdown(
                                TrainerDataBuilderConfig.model_fields["output_template"].description,
                            ),
                            rx.dialog.close(rx.button("Close", mt="4")),
                            size="4",
                        ),
                    ),
                    width="100%",
                ),
                rx.text_area(
                    placeholder="",
                    name="output_template",
                    value=AppState.output_template,
                    on_change=AppState.setvar("output_template"),
                    size="3",
                    width="100%",
                    rows="5",
                ),
                rx.button(
                    "Next",
                    type="submit",
                    width="100%",
                    size="3",
                    variant="solid",
                ),
                width="100%",
                align_items="column",
            ),
            width="60vw",
            margin_top="4",
        ),
        spacing="5",
        padding="4",
        width="60vw",
        on_submit=AppState.handle_prompts_submit,
    )

    jobs_table = rx.vstack(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    _header_cell("Priority", "gauge"),
                    _header_cell("Mode", "beaker"),
                    _header_cell("Run Name", "fingerprint"),
                    # _header_cell("Est. Runtime", "hourglass"),
                    # _header_cell("Est. Co2 Emission", "leaf"),
                    _header_cell("Status", "cog"),
                ),
            ),
            rx.table.body(rx.foreach(AppState.configurator_outputs, show_configuration)),
            variant="surface",
            size="3",
            width="100%",
        ),
        execute_configs_dialog(),
        align="center",
        spacing="2",
    )

    results_tab = rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("layout-template", size=20),
                rx.text("Results", weight="bold"),
            ),
            rx.el.iframe(
                src_doc=ConfigurationState.result_fig,
                width="100%",
                height="100%",
            ),
            width="100%",
            height="100%",
        ),
        on_mount=ConfigurationState.load_config_group_results(AppState.run_group),
        spacing="2",
        align="center",
    )

    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Settings", value="settings"),
            rx.tabs.trigger("Models", value="models"),
            rx.tabs.trigger("Prompts", value="prompts"),
            rx.tabs.trigger("Validate", value="validate"),
            rx.tabs.trigger("Results", value="results"),
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
        rx.tabs.content(
            results_tab,
            value="results",
        ),
        default_value="settings",
        value=AppState.current_tab,
        on_change=AppState.set_current_tab,
    )
