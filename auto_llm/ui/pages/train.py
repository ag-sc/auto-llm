import json
import subprocess

import gradio as gr
import pydantic_core
import wandb
import yaml

from auto_llm.dto.example_dtos import EXAMPLE_TRAINER_RUN_CONFIG
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.registry.configurator_registry import TRAINER_RUN_SCRIPT

CONFIG_ROOT_PATH = "/vol/auto_llm/config_files/trainer_configs"


def save_trainer_run_config(
    uploaded_config: str,
    uploaded_config_path: str,
    defined_config: str,
    defined_config_path: str,
) -> str:
    if uploaded_config:
        gr.Success(
            f"The config you uploaded from the path: '{uploaded_config_path}' will be used for training!"
        )
        config_path = uploaded_config_path
    elif defined_config:
        with open(defined_config_path, "w+") as f:
            yaml.dump(yaml.safe_load(defined_config), f)
        gr.Success(
            f"The config you defined is saved to '{defined_config_path}' and will be used for training!"
        )
        config_path = defined_config_path
    else:
        raise gr.Error(
            f"Please upload or define a Trainer Run Configuration to proceed!"
        )

    return config_path


def start_trainer_run(config_path: str, venv_path: str, env_path: str):
    info = subprocess.check_output(
        [
            "sbatch",
            TRAINER_RUN_SCRIPT,
            config_path,
            venv_path,
            env_path,
        ]
    )
    info = info.decode("utf-8")
    gr.Info(info)


def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


def get_trainer_run_config(config_path: str = None) -> str:
    if config_path:
        config = validate_config(config_path=config_path)
    else:
        config = EXAMPLE_TRAINER_RUN_CONFIG

    config = config.model_dump_json()
    config_yaml = yaml.dump(json.loads(config))
    return config_yaml


def validate_config(config_path: str = None, config_text: str = None):
    try:
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        if config_text:
            config = yaml.safe_load(config_text)

        config = TrainerRunConfig.model_validate(config)
    except UnicodeDecodeError as e:
        raise gr.Error(
            title="Corrupted YAML",
            message=f"Please check the config file. Your YAML file may be corrupted. See error below.<br><br>{e}",
        )
    except pydantic_core._pydantic_core.ValidationError as e:
        raise gr.Error(
            title="Validation Error",
            message=f"Please check the config file. Your YAML file is not of the expected configuration format. See error below.<br><br>{e}",
        )

    return config


def validate_defined_config(config_text: str, defined_config_path: str):
    if config_text != "":
        config = validate_config(config_text=config_text)
        if defined_config_path == "":
            raise gr.Error(f"Please provide a path to save the defined config file!")


def set_and_unset_config(config_to_set, config_to_unset):
    return config_to_set, ""


with gr.Blocks() as demo:
    # build trainer run config
    with gr.Tab("Define"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "**Option 1:** Upload a valid trainer run configuration here."
                )
                uploaded_config_path = gr.UploadButton(label="Upload ⚙️")
                uploaded_config = gr.Code(
                    value="",
                    language="yaml",
                    interactive=False,
                    show_label=False,
                    lines=10,
                )
                uploaded_config_path_text = gr.Textbox(
                    label="Uploaded configuration file path", interactive=False
                )
            with gr.Column():
                gr.Markdown("**Option 2:** Define the trainer run configuration here.")
                define_btn = gr.Button(value="Define ⚙️")
                defined_config = gr.Code(
                    value="",
                    language="yaml",
                    interactive=True,
                    show_label=False,
                    lines=13,
                )
                defined_config_path = gr.Textbox(
                    label="Path to save the configuration file"
                )

        btn = gr.Button("Submit")
    with gr.Tab("Train"):
        gr.Markdown("## Training")
        report_url = "https://api.wandb.ai/links/llm4kmu/v9qfir8d"
        report = wandb_report(report_url)

    config_path = gr.State()

    # TODO: How to handle these? common venv in cluster?
    #  env.sh should also be set.
    venv_path = gr.State("venv")
    env_path = gr.State("../env.sh")

    uploaded_config_path.click(fn=lambda x: x, inputs=[], outputs=[defined_config_path])

    uploaded_config_path.upload(
        fn=set_and_unset_config,
        inputs=[uploaded_config, defined_config],
        outputs=[uploaded_config, defined_config],
        show_progress="hidden",
    ).then(
        fn=lambda x: x,
        inputs=[],
        outputs=[defined_config_path],
        show_progress="hidden",
    ).then(
        fn=get_trainer_run_config,
        inputs=[uploaded_config_path],
        outputs=[uploaded_config],
        show_progress="hidden",
    ).success(
        fn=lambda x: x,
        inputs=[uploaded_config_path],
        outputs=[uploaded_config_path_text],
        show_progress="hidden",
    )

    define_btn.click(
        fn=set_and_unset_config,
        inputs=[defined_config, uploaded_config],
        outputs=[defined_config, uploaded_config],
        show_progress="hidden",
    ).then(
        fn=lambda x: x,
        inputs=[],
        outputs=[uploaded_config_path_text],
        show_progress="hidden",
    ).then(
        fn=get_trainer_run_config,
        inputs=[],
        outputs=[defined_config],
        show_progress="hidden",
    )

    btn.click(
        fn=validate_defined_config,
        inputs=[defined_config, defined_config_path],
        outputs=[],
        show_progress="hidden",
    ).success(
        fn=save_trainer_run_config,
        inputs=[
            uploaded_config,
            uploaded_config_path,
            defined_config,
            defined_config_path,
        ],
        outputs=[config_path],
        show_progress="hidden",
    ).then(
        fn=start_trainer_run,
        inputs=[config_path, venv_path, env_path],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch()
    demo.integrate(wandb=wandb)
