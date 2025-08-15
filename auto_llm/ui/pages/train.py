import json
import os

import gradio as gr
import pydantic_core
import wandb
import yaml

from auto_llm.dto.builder_config import TrainerDataBuilderConfig
from auto_llm.dto.trainer_run_config import (
    TrainerRunConfig,
    AutoLlmTrainerArgs,
    TrainerArgs,
    LoraConfig,
)

EXAMPLE_TRAINER_RUN_CONFIG = TrainerRunConfig(
    auto_llm_trainer_args=AutoLlmTrainerArgs(
        trainer_type="sft",
        model_name="google/gemma-2-2b-it",
        truncation=True,
        completion_only_loss=True,
    ),
    trainer_args=TrainerArgs(
        max_length=1024,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=True,
        auto_find_batch_size=False,
        output_dir="",
        resume_from_checkpoint=None,
        gradient_checkpointing=True,
        num_train_epochs=10,
        learning_rate=5e-5,
        weight_decay=0,
        lr_scheduler_type="linear",
        warmup_steps=0,
        report_to="wandb",
        run_name="some name",
        logging_steps=0.1,
        eval_strategy="steps",
        bf16_full_eval=False,
        fp16_full_eval=False,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=None,
    ),
    trainer_data_builder_config=TrainerDataBuilderConfig(
        dataset_dir="",
        instruction_template="",
        input_template="",
        output_template="",
        dataset_type="conversational",
        instruction_input_separator="\n",
        use_system_message=True,
        parse_output_as_json=True,
        num_few_shot_examples=None,
        few_shot_examples_split="validation",
    ),
    peft_config=LoraConfig(),
)


def save_trainer_run_config(
    uploaded_config,
    uploaded_config_path,
    defined_config,
    defined_config_path,
) -> str:
    if uploaded_config:
        gr.Info(
            f"The config you uploaded from the path: '{uploaded_config_path}' will be used for training!"
        )
        config_path = uploaded_config_path
    elif defined_config:
        if defined_config_path == "":
            raise gr.Error(f"Please provide a path to save the defined config file!")
        with open(defined_config_path, "w+") as f:
            yaml.dump(yaml.safe_load(defined_config), f)
        gr.Info(
            f"The config you defined is saved to '{defined_config_path}' and will be used for training!"
        )
        config_path = defined_config_path
    else:
        raise gr.Error(
            f"Please upload or define a Trainer Run Configuration to proceed!"
        )

    return config_path


def start_trainer_run(config_path: str, venv_path: str, env_path: str):
    cmd = f"sbatch scripts/autollm_train.sbatch {config_path} {venv_path} {env_path}"
    os.system(cmd)


def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


def get_trainer_run_config(config_path: str = None) -> str:
    if config_path:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = TrainerRunConfig.model_validate(config)
            gr.Success(f"The YAML you uploaded is validated!")
        except UnicodeDecodeError as e:
            raise gr.Error(
                f"Please check the config file. Your YAML file may be corrupted. Error: {e}"
            )
        except pydantic_core._pydantic_core.ValidationError as e:
            raise gr.Error(
                f"Please check the config file. Your YAML file is not of the expected configuration format. Error: {e}"
            )

    else:
        config = EXAMPLE_TRAINER_RUN_CONFIG

    config = config.model_dump_json()
    config_yaml = yaml.dump(json.loads(config))
    return config_yaml


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
                    label="Uploaded file path", interactive=False
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
                defined_config_path = gr.Textbox(label="Path to save the config file")

        btn = gr.Button("Submit")
    with gr.Tab("Train"):
        gr.Markdown("## Training")
        report_url = "https://api.wandb.ai/links/llm4kmu/v9qfir8d"
        report = wandb_report(report_url)

    config_path = gr.State()
    venv_path = gr.State("venv")
    env_path = gr.State("../env.sh")

    uploaded_config_path.upload(
        fn=set_and_unset_config,
        inputs=[uploaded_config, defined_config],
        outputs=[uploaded_config, defined_config],
    ).then(
        fn=get_trainer_run_config,
        inputs=[uploaded_config_path],
        outputs=[uploaded_config],
    ).success(
        fn=lambda x: x,
        inputs=[uploaded_config_path],
        outputs=[uploaded_config_path_text],
    )

    define_btn.click(
        fn=set_and_unset_config,
        inputs=[defined_config, uploaded_config],
        outputs=[defined_config, uploaded_config],
    ).then(
        fn=lambda x: x,
        inputs=[],
        outputs=[uploaded_config_path_text],
    ).then(
        fn=get_trainer_run_config,
        inputs=[],
        outputs=[defined_config],
    )

    btn.click(
        fn=save_trainer_run_config,
        inputs=[
            uploaded_config,
            uploaded_config_path,
            defined_config,
            defined_config_path,
        ],
        outputs=[config_path],
    ).then(
        fn=start_trainer_run,
        inputs=[config_path, venv_path, env_path],
    )


if __name__ == "__main__":
    demo.launch()
    demo.integrate(wandb=wandb)
