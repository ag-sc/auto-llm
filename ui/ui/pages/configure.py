import pandas as pd
import reflex as rx

from auto_llm.configurator.config_generator import ConfiguratorOutput, ConfigMode, Priority
from auto_llm.dto.builder_config import TrainerDataBuilderConfig
from auto_llm.estimator.utils import get_gpu_params
from auto_llm.registry.tracker_registry import WANDB_TRAIN_PROJECT, WANDB_EVAL_PROJECT
from auto_llm.tasks.registry import TASKS

from ..state.app_state import AppState
from ..state.user import User
from ..state.configuration_state import ConfigurationState, InsightRun, MetricInsight, RunInfoItem, RunStat
from .. import styles

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


def example_result_card(result) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text(result.model_label, weight="bold", size="2"),
                    rx.text(result.run_name, size="1", color_scheme="gray", trim="both"),
                    spacing="1",
                    min_width="0",
                ),
                rx.spacer(),
                rx.hstack(
                    rx.foreach(
                        result.scores,
                        lambda score: rx.badge(score.name, ": ", score.value, color_scheme=score.color, variant="soft", size="2"),
                    ),
                    spacing="2",
                    wrap="wrap",
                    justify="end",
                ),
                width="100%",
                align="start",
            ),
            rx.box(
                rx.text(result.output, size="2", white_space="pre-wrap", word_break="break-word"),
                background_color=rx.color("gray", 2),
                border_radius="var(--radius-2)",
                padding="0.5em 0.75em",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
    )


def examples_popover():
    example = ConfigurationState.current_example

    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                "View Examples",
                variant="soft",
                size="3",
                disabled=rx.cond(
                    ConfigurationState.is_loading_results,  # False
                    True,
                    rx.cond(
                        ConfigurationState.is_results_loaded,  # True
                        False,
                        True,
                    ),
                ),
                loading=ConfigurationState.is_loading_results,
            )
        ),
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.dialog.title("Examples", margin="0"),
                    rx.spacer(),
                    rx.cond(
                        ConfigurationState.num_examples > 0,
                        rx.hstack(
                            rx.icon_button(
                                rx.icon("chevron-left", size=16),
                                variant="soft",
                                size="1",
                                on_click=ConfigurationState.prev_example,
                                disabled=ConfigurationState.example_index == 0,
                            ),
                            rx.text(
                                ConfigurationState.example_index + 1,
                                " / ",
                                ConfigurationState.num_examples,
                                size="2",
                                color_scheme="gray",
                            ),
                            rx.icon_button(
                                rx.icon("chevron-right", size=16),
                                variant="soft",
                                size="1",
                                on_click=ConfigurationState.next_example,
                                disabled=ConfigurationState.example_index >= ConfigurationState.num_examples - 1,
                            ),
                            align="center",
                            spacing="2",
                        ),
                    ),
                    width="100%",
                    align="center",
                ),
                rx.cond(
                    ConfigurationState.num_examples > 0,
                    rx.box(
                        rx.vstack(
                            rx.text("Input", weight="bold", size="2", color_scheme="gray"),
                            rx.box(
                                rx.text(example.input_text, size="2", white_space="pre-wrap", word_break="break-word"),
                                background_color=rx.color("gray", 2),
                                border=f"1px solid {rx.color('gray', 5)}",
                                border_radius="var(--radius-3)",
                                padding="0.75em 1em",
                                max_height="30vh",
                                overflow_y="auto",
                                width="100%",
                            ),
                            rx.text("Expected output", weight="bold", size="2", color_scheme="gray", padding_top="0.5em"),
                            rx.box(
                                rx.text(example.expected_output, size="2", white_space="pre-wrap", word_break="break-word", color=rx.color("blue", 11)),
                                background_color=rx.color("blue", 2),
                                border=f"1px solid {rx.color('blue', 6)}",
                                border_radius="var(--radius-3)",
                                padding="0.75em 1em",
                                width="100%",
                            ),
                            rx.hstack(
                                rx.text("Model outputs", weight="bold", size="2", color_scheme="gray"),
                                rx.spacer(),
                                rx.hstack(
                                    rx.badge("best", color_scheme="green", variant="soft"),
                                    rx.badge("middle", color_scheme="amber", variant="soft"),
                                    rx.badge("worst", color_scheme="red", variant="soft"),
                                    rx.badge("tied", color_scheme="gray", variant="soft"),
                                    rx.text("per metric, across models", size="1", color_scheme="gray"),
                                    spacing="1",
                                    align="center",
                                    wrap="wrap",
                                ),
                                width="100%",
                                align="center",
                                padding_top="0.5em",
                            ),
                            rx.foreach(example.results, example_result_card),
                            spacing="2",
                            width="100%",
                            padding_right="1em",
                        ),
                        max_height="65vh",
                        overflow_y="auto",
                        overflow_x="hidden",
                        width="100%",
                    ),
                    rx.callout("No examples available for this run group.", icon="info", width="100%"),
                ),
                rx.hstack(
                    rx.dialog.close(rx.button("Close", variant="soft")),
                    width="100%",
                    justify="end",
                ),
                spacing="4",
                width="100%",
            ),
            max_width="min(900px, 95vw)",
        ),
    )


def _insight_run_row(run: InsightRun):
    """One model's score for a metric: model kind, model name, a bar relative to the top score, and the value."""
    return rx.hstack(
        rx.tooltip(
            rx.badge(
                run.kind_label,
                color_scheme=rx.match(run.kind, ("pre", "gray"), ("fft", "blue"), ("lora", "violet"), "gray"),
                variant="soft",
                min_width="9.5em",
                justify="center",
            ),
            content=run.run_name,
        ),
        rx.text(run.model, size="2", color_scheme="gray", min_width="11em", white_space="nowrap"),
        rx.progress(value=run.bar, color_scheme=rx.cond(run.is_best, "green", "gray"), size="2", flex="1"),
        rx.text(
            run.value,
            size="2",
            weight=rx.cond(run.is_best, "bold", "regular"),
            font_family="monospace",
            min_width="3.5em",
            text_align="right",
        ),
        rx.cond(run.is_best, rx.icon("trophy", size=14, color=rx.color("amber", 9)), rx.box(width="14px")),
        width="100%",
        align="center",
        spacing="3",
    )


def _metric_insight_card(insight: MetricInsight):
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("gauge", size=16, color=rx.color("accent", 10)),
                    rx.code(insight.metric, variant="soft", size="3"),
                    align="center",
                    spacing="2",
                ),
                rx.spacer(),
                rx.vstack(
                    rx.text(insight.best_value, size="5", weight="bold", font_family="monospace", line_height="1"),
                    rx.text("best · ", insight.direction, size="1", color_scheme="gray"),
                    spacing="1",
                    align="end",
                ),
                width="100%",
                align="center",
            ),
            rx.vstack(rx.foreach(insight.runs, _insight_run_row), width="100%", spacing="2"),
            rx.callout.root(
                rx.callout.icon(
                    rx.match(
                        insight.tone,
                        ("green", rx.icon("trending-up", size=16)),
                        ("amber", rx.icon("triangle-alert", size=16)),
                        rx.icon("equal", size=16),
                    )
                ),
                rx.callout.text(insight.verdict, size="2"),
                color_scheme=insight.tone,
                variant="surface",
                size="1",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        width="100%",
    )


def explanation_popover():
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                "Interpret Results",
                variant="soft",
                size="3",
                disabled=rx.cond(
                    ConfigurationState.is_loading_results,  # False
                    True,
                    rx.cond(
                        ConfigurationState.is_results_loaded,  # True
                        False,
                        True,
                    ),
                ),
                loading=ConfigurationState.is_loading_results,
            )
        ),
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.icon("badge-check", size=22, color=rx.color("accent", 10)),
                    rx.dialog.title("How to interpret the results?", margin="0"),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    "For each metric, the bars compare every evaluated model against the best score. The best model is marked with a trophy.",
                    size="2",
                    color_scheme="gray",
                ),
                rx.scroll_area(
                    rx.cond(
                        ConfigurationState.result_insights.length() > 0,
                        rx.vstack(rx.foreach(ConfigurationState.result_insights, _metric_insight_card), spacing="3", width="100%"),
                        rx.text("No evaluation metrics found for this run group yet.", color_scheme="gray"),
                    ),
                    type="auto",
                    scrollbars="vertical",
                    max_height="60vh",
                    padding_right="3",
                ),
                rx.callout.root(
                    rx.callout.icon(rx.icon("book-open", size=16)),
                    rx.callout.text(
                        "Please refer to the ",
                        rx.link("docs", href="https://ag-sc.github.io/auto-llm/fundamentals/evaluating/", is_external=True),
                        " for a detailed explanation of the evaluation metrics.",
                    ),
                    color_scheme="gray",
                    variant="soft",
                    size="1",
                    width="100%",
                ),
                rx.hstack(rx.dialog.close(rx.button("Close", variant="soft")), justify="end", width="100%"),
                spacing="4",
                width="100%",
            ),
            max_width="min(900px, 95vw)",
            size="4",
        ),
    )


def _run_state_badge(state):
    return rx.match(
        state,
        ("running", status_badge("running")),
        ("finished", status_badge("finished")),
        ("failed", status_badge("failed")),
        ("crashed", status_badge("crashed")),
        ("killed", status_badge("killed")),
        rx.fragment(),
    )


def _run_info_item(item: RunInfoItem):
    return rx.vstack(
        rx.text(item.label, size="1", color_scheme="gray"),
        rx.text(item.value, size="2", weight="medium", word_break="break-all"),
        spacing="0",
    )


def _run_stat_card(stat: RunStat):
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.cond(stat.section != "", rx.text(stat.section, size="1", color_scheme="gray")),
                rx.code(stat.name, variant="ghost", size="2"),
                spacing="1",
                align="center",
            ),
            rx.hstack(
                rx.text(stat.value, size="5", weight="bold", font_family="monospace", line_height="1.2"),
                rx.text(stat.stderr, size="1", color_scheme="gray"),
                align="baseline",
                spacing="2",
            ),
            spacing="1",
        ),
        size="1",
    )


def run_view_dialog(configurator_output: ConfiguratorOutput):
    """In-app overview of the W&B run: drawn from the W&B API, so viewers need no W&B login."""
    return rx.dialog.root(
        rx.tooltip(
            rx.dialog.trigger(
                rx.icon_button(
                    "monitor",
                    variant="soft",
                    size="1",
                    on_click=ConfigurationState.load_run_view(configurator_output.run_id, configurator_output.run_name, configurator_output.mode),
                ),
            ),
            content="View run",
        ),
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.icon("activity", size=22, color=rx.color("accent", 10)),
                    rx.dialog.title("Run overview", margin="0"),
                    _run_state_badge(ConfigurationState.run_view_state),
                    rx.spacer(),
                    rx.tooltip(
                        rx.icon_button(
                            "refresh-cw",
                            variant="soft",
                            size="2",
                            on_click=ConfigurationState.refresh_run_view,
                            loading=ConfigurationState.is_loading_run_view,
                        ),
                        content="Refresh",
                    ),
                    rx.cond(
                        ConfigurationState.run_view_url != "",
                        rx.link(
                            rx.button(rx.icon("external-link", size=16), "Open in W&B", variant="soft", size="2"),
                            href=ConfigurationState.run_view_url,
                            is_external=True,
                        ),
                    ),
                    align="center",
                    spacing="3",
                    width="100%",
                ),
                rx.dialog.description(ConfigurationState.run_view_name, size="2", color_scheme="gray", word_break="break-all"),
                rx.cond(
                    ConfigurationState.is_loading_run_view,
                    rx.center(rx.spinner(size="3"), height="30vh", width="100%"),
                    rx.cond(
                        ConfigurationState.run_view_error != "",
                        rx.callout(ConfigurationState.run_view_error, icon="info", color_scheme="gray", width="100%"),
                        rx.vstack(
                            rx.card(
                                rx.grid(
                                    rx.foreach(ConfigurationState.run_view_info, _run_info_item),
                                    columns=rx.breakpoints(initial="2", sm="3", md="5"),
                                    spacing="4",
                                    width="100%",
                                ),
                                width="100%",
                            ),
                            rx.cond(
                                ConfigurationState.run_view_stats.length() > 0,
                                rx.grid(
                                    rx.foreach(ConfigurationState.run_view_stats, _run_stat_card),
                                    columns=rx.breakpoints(initial="2", sm="3", md="4"),
                                    spacing="3",
                                    width="100%",
                                ),
                            ),
                            rx.cond(
                                ConfigurationState.run_view_has_chart,
                                rx.card(
                                    rx.plotly(data=ConfigurationState.run_view_fig, use_resize_handler=True, width="100%"),
                                    width="100%",
                                ),
                            ),
                            spacing="3",
                            width="100%",
                        ),
                    ),
                ),
                rx.hstack(rx.dialog.close(rx.button("Close", variant="soft")), justify="end", width="100%"),
                spacing="3",
                width="100%",
            ),
            max_width="min(1100px, 95vw)",
            max_height="90vh",
            overflow_y="auto",
            size="4",
        ),
    )


def wandb_run_link(configurator_output: ConfiguratorOutput):
    """Opens the run on wandb.ai in a new tab (built client-side, no backend round trip)."""
    project_name = rx.cond(configurator_output.mode == ConfigMode.TRAINER_RUN_CFG, WANDB_TRAIN_PROJECT, WANDB_EVAL_PROJECT)
    return rx.tooltip(
        rx.link(
            rx.icon_button("external-link", variant="soft", size="1"),
            href=f"https://wandb.ai/{User.wandb_entity}/{project_name}/runs/{configurator_output.run_id}",
            is_external=True,
        ),
        content="Open in W&B",
    )


def config_view_dialog(configurator_output: ConfiguratorOutput):
    configuration_description = f"""The run configuration is detailed below. You can customize these paramters or proceed with the default values.

ℹ️ Please refer to the documentation for a detailed guide on the configuration parameters.
"""
    return rx.dialog.root(
        rx.tooltip(
            rx.dialog.trigger(
                rx.icon_button(
                    "pencil",
                    variant="soft",
                    size="1",
                    on_click=lambda: ConfigurationState.load_config(configurator_output),
                ),
            ),
            content="Edit Configuration",
        ),
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title(f"Configuration"),
                rx.markdown(configuration_description),
                rx.text_area(
                    value=ConfigurationState.current_yaml_content,
                    on_change=ConfigurationState.update_content,
                    font_family="Source Code Pro, Menlo, Monaco, Lucide Console, monospace",
                    font_size="13px",
                    line_height="1.5",
                    width="100%",
                    height="450px",
                    style={
                        "white-space": "pre",
                        "overflow_x": "auto",
                        "tab_size": "2",
                        "resize": "vertical",
                        "padding": "1rem",
                        "border": "1px solid var(--gray-5)",
                        "background": "var(--gray-2)",
                    },
                ),
                rx.hstack(
                    rx.dialog.close(rx.button("Close", variant="soft")),
                    rx.button("Save", on_click=ConfigurationState.save_config),
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
            rx.dialog.trigger(rx.button("Execute", variant="soft", size="3", on_click=AppState.handle_validation_submit_pre)),
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
                (Priority.PRIORITY_ONE, status_badge("1")),
                (Priority.PRIORITY_TWO, status_badge("2")),
                (Priority.PRIORITY_THREE, status_badge("3")),
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
        rx.table.cell(rx.text(configurator_output.run_name)),
        rx.table.cell(
            rx.hstack(
                config_view_dialog(configurator_output),
                run_view_dialog(configurator_output),
                wandb_run_link(configurator_output),
                rx.match(
                    current_status,
                    ("running", status_badge("running")),
                    ("finished", status_badge("finished")),
                    ("failed", status_badge("failed")),
                    ("crashed", status_badge("crashed")),
                    ("killed", status_badge("killed")),
                    ("pending", status_badge("pending")),
                ),
            ),
            align="center",
        ),
        style={"_hover": {"bg": rx.color("gray", 3)}},
        align="center",
    )


@template(
    route="/configure",
    title="Configure",
    on_load=[
        User.check_logged_in,
    ],
)
def configure() -> rx.Component:
    dataset_path_descr = f"""
    Path of the dataset. We currently support remote HuggingFace Datasets paths. 

    Examples:
    {AppState.dataset_options_markdown}
    """

    tasks_descr = """
    Category of the task. This can either be:
    - **Sequence To Sequence**
    - **Sequence To Label**
    - **Sequence To Structured Output**

    Please read the documentation [here](https://ag-sc.github.io/auto-llm/fundamentals/tasks/) for a detailed overview of the supported tasks.
    """

    form = rx.form(
        rx.card(
            rx.vstack(
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
                rx.grid(
                    rx.vstack(
                        rx.hstack(rx.icon("microchip", size=20), rx.text("Hardware Type", weight="bold")),
                        rx.select(
                            GPU_PARAMS,
                            placeholder="Select GPU...",
                            name="hardware_type",
                            value=AppState.hardware_type,
                            width="100%",
                            size="3",
                            disabled=True,
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
                            width="100%",
                            size="3",
                            disabled=True,
                        ),
                        align_items="start",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                rx.hstack(
                    rx.button(
                        "Reset",
                        type="button",
                        width="50%",
                        size="3",
                        variant="outline",
                        on_click=AppState.reset_state,
                    ),
                    rx.button(
                        "Next",
                        type="submit",
                        width="50%",
                        size="3",
                        variant="solid",
                    ),
                    align="center",
                    width="100%",
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
                            width="100%",
                            height="100%",
                        ),
                        width="100%",
                        height="100%",
                    ),
                    width="100%",
                ),
                rx.box(
                    rx.foreach(
                        AppState.model_choices,
                        lambda choice: rx.checkbox(
                            choice,
                            on_change=lambda checked: AppState.toggle_choice(choice, checked),
                            checked=AppState.selected_models.contains(choice),
                            spacing="2",
                        ),
                    ),
                ),
                rx.hstack(
                    rx.icon("package-plus", size=20),
                    rx.text("Custom Models", weight="bold"),
                    width="100%",
                    margin_top="2",
                ),
                rx.text(
                    "Add open-source models from HuggingFace of your choice. Use the full name of the model e.g. meta-llama/Llama-3.2-3B-Instruct. "
                    "You can use comma-separated list for more models.",
                    size="2",
                    color_scheme="gray",
                ),
                rx.input(
                    placeholder="meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen2.5-7B-Instruct",
                    name="custom_models",
                    value=AppState.custom_models,
                    on_change=AppState.setvar("custom_models"),
                    size="3",
                    width="100%",
                ),
                rx.hstack(
                    rx.button(
                        "Reset",
                        type="button",
                        width="50%",
                        size="3",
                        variant="outline",
                        on_click=AppState.reset_state,
                    ),
                    rx.button(
                        "Next",
                        type="submit",
                        width="50%",
                        size="3",
                        variant="solid",
                        loading=AppState.is_checking_models,
                    ),
                    align="center",
                    width="100%",
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
                rx.hstack(
                    rx.button(
                        "Reset",
                        type="button",
                        width="50%",
                        size="3",
                        variant="outline",
                        on_click=AppState.reset_state,
                    ),
                    rx.button(
                        "Next",
                        type="submit",
                        width="50%",
                        size="3",
                        variant="solid",
                    ),
                    align="center",
                    width="100%",
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
        on_submit=lambda: AppState.handle_prompts_submit,
    )

    jobs_table = rx.vstack(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    _header_cell("Priority", "gauge"),
                    _header_cell("Mode", "beaker"),
                    _header_cell("Run Name", "fingerprint"),
                    _header_cell("Details", "cog"),
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
        on_mount=ConfigurationState.start_polling,  # Polling is now securely initialized once here instead of flooding row events
        on_unmount=ConfigurationState.stop_polling,
    )

    results_tab = rx.card(
        rx.vstack(
            rx.hstack(
                rx.hstack(rx.icon("folder-kanban", size=25), rx.heading("Results", size="5", weight="bold"), align="center"),
                rx.button(
                    rx.icon(
                        tag="refresh-cw",
                        size=20,
                    ),
                    variant="soft",
                    on_click=ConfigurationState.load_config_group_results(AppState.run_group),
                    loading=ConfigurationState.is_loading_results,
                ),
                width="100%",
                justify="between",
                align="center",
            ),
            rx.card(
                rx.el.iframe(
                    src_doc=ConfigurationState.result_fig,
                    width="100%",
                    height="100%",
                    style={"border": "none", "display": "block", "overflow": "hidden"},
                    loading="lazy",
                ),
                size="3",
                box_shadow=styles.box_shadow_style,
                width="100%",
                height="60%",
            ),
            rx.box(height="5%"),
            rx.box(
                rx.hstack(
                    explanation_popover(),
                    examples_popover(),
                    align="center",
                    spacing="5",
                    justify="center",
                ),
                size="3",
                padding="4",
                width="100%",
                height="10%",
            ),
            width="100%",
            height="100%",
        ),
        on_mount=ConfigurationState.reset_results_state,
        spacing="5",
        size="3",
        width="60vw",
        height="80vh",
    )

    return rx.flex(
        rx.box(
            rx.tabs.root(
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
            ),
        ),
        width="100%",
        height="100%",
        direction={"sm": "row", "md": "row"},
    )
