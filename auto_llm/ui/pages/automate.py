from typing import List

import gradio as gr
import yaml

from auto_llm.builder.task_data_builder.registry import (
    INSTRUCTION_TEMPLATES_MAPPING,
    INPUT_TEMPLATES_MAPPING,
    OUTPUT_TEMPLATES_MAPPING,
)
from auto_llm.configurator.config_generator import (
    MODEL_NAMES,
    TrainEvalRunConfigurator,
    TASKS,
)
from auto_llm.dto.builder_config import TrainerDataBuilderConfig


def generate_configs(
    model_names: List[str],
    task: str,
    dataset_name: str,
    dataset_dir: str,
    output_path: str,
    configs_path: str,
    instruction_template: str,
    input_template: str,
    output_template: str,
):
    incomplete = False
    for var_value in locals().values():
        if var_value in ("", None, []):
            incomplete = True
            break

    if incomplete:
        raise gr.Error("Please fill all fields to proceed!")

    configurator = TrainEvalRunConfigurator(
        model_names=model_names,
        task=task,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        output_path=output_path,
        configs_path=configs_path,
        instruction_template=instruction_template,
        input_template=input_template,
        output_template=output_template,
    )

    configurator.generate()


def update_file_explorer(configs_path: str):
    return gr.update(root_dir=configs_path, interactive=True, visible=True)


def view_yaml_file(path: str):
    if not path:
        return None

    with open(path) as f:
        data = yaml.safe_load(f)

    data = yaml.dump(data)
    return data


def update_file_name(path: str):
    if not path:
        return gr.update(label="", interactive=True, visible=True)
    return gr.update(label=path.split("/")[-1], interactive=True, visible=True)


def update_save_btn():
    return gr.update(visible=True)


def update_instruction_textbox(task: str):
    return INSTRUCTION_TEMPLATES_MAPPING.get(task)


def update_input_textbox(task: str):
    return INPUT_TEMPLATES_MAPPING.get(task)


def update_output_textbox(task: str):
    return OUTPUT_TEMPLATES_MAPPING.get(task)


def save_file(file_path: str, file_contents: str):
    try:
        with open(file_path, "w+") as f:
            yaml.safe_dump(yaml.safe_load(file_contents), f, indent=4)
        gr.Info(f"Saved file: {file_path}")
    except:
        raise gr.Error(f"YAML file is corrupt. Cannot save file: {file_path}")


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            task = gr.Dropdown(
                label="Task Name",
                choices=TASKS,
                value=None,  # noqa
                multiselect=False,
                allow_custom_value=True,
                interactive=True,
                info=f"Name of the task you configured.",
            )

    with gr.Row(equal_height=True):
        with gr.Column():
            dataset_name = gr.Textbox(
                label="Dataset Name",
                info=f"Name of the dataset belonging to the ``Task`` you configured.",
            )
        with gr.Column():
            dataset_dir = gr.Textbox(
                label="Dataset Directory",
                info=TrainerDataBuilderConfig.model_fields["dataset_dir"].description,
            )

    with gr.Row(equal_height=True):
        with gr.Column():
            model_names = gr.Dropdown(
                label="Models",
                choices=MODEL_NAMES,
                multiselect=True,
                allow_custom_value=True,
                info="Select models oy our choice",
            )
        with gr.Column():
            output_path = gr.Textbox(
                label="Output Path",
                info="Path where the trained moels should be saved",
            )

    with gr.Row(equal_height=True):
        with gr.Column():
            instruction_template = gr.Textbox(
                label="Instruction Template",
                lines=10,
                info=TrainerDataBuilderConfig.model_fields[
                    "instruction_template"
                ].description,
                # value=INSTRUCTION_TEMPLATES_MAPPING.get(task),
            )
        with gr.Column():
            input_template = gr.Textbox(
                label="Input Template",
                lines=10,
                info=TrainerDataBuilderConfig.model_fields[
                    "input_template"
                ].description,
                # value=INPUT_TEMPLATES_MAPPING.get(task),
            )
        with gr.Column():
            output_template = gr.Textbox(
                label="Output Template",
                lines=10,
                info=TrainerDataBuilderConfig.model_fields[
                    "output_template"
                ].description,
                # value=OUTPUT_TEMPLATES_MAPPING.get(task),
            )

    with gr.Row():
        configs_path = gr.Textbox(
            label="Configs Path",
            info="Path where the trainer/evaluator configurations should be saved",
        )

    submit_btn = gr.Button("Generate configs")

    with gr.Row():
        with gr.Column():
            config_explorer = gr.FileExplorer(
                glob="*.yaml", file_count="single", visible=False
            )
        with gr.Column():
            file_viewer = gr.Code(label="", language="yaml", visible=False)
            save_btn = gr.Button(value="Save file", visible=False)

    task.input(
        fn=update_instruction_textbox,
        inputs=[task],
        outputs=[instruction_template],
    ).then(
        fn=update_input_textbox,
        inputs=[task],
        outputs=[input_template],
    ).then(
        fn=update_output_textbox,
        inputs=[task],
        outputs=[output_template],
    )

    submit_btn.click(
        fn=generate_configs,
        inputs=[
            model_names,
            task,
            dataset_name,
            dataset_dir,
            output_path,
            configs_path,
            instruction_template,
            input_template,
            output_template,
        ],
    ).success(fn=update_file_explorer, inputs=[configs_path], outputs=[config_explorer])

    config_explorer.change(
        fn=view_yaml_file, inputs=[config_explorer], outputs=[file_viewer]
    ).success(
        fn=update_file_name, inputs=[config_explorer], outputs=[file_viewer]
    ).then(
        fn=update_save_btn, inputs=[], outputs=[save_btn]
    )

    save_btn.click(
        fn=save_file,
        inputs=[config_explorer, file_viewer],
    )

if __name__ == "__main__":
    demo.launch()
