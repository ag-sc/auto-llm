import datetime
import json
import subprocess
from typing import List

import gradio as gr
import pandas as pd
import yaml

from auto_llm.automator.automator import Automator

from auto_llm.configurator.config_executor import ConfigExecutor, SequentialConfigExecutor
from auto_llm.configurator.config_generator import (
    TrainEvalRunConfigurator,
    ConfiguratorOutput,
)

from auto_llm.dto.builder_config import TrainerDataBuilderConfig
from auto_llm.estimator.emission_estimator import EmissionEstimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator
from auto_llm.estimator.utils import get_gpu_params, get_model_params
from auto_llm.registry.tracker_registry import (
    WANDB_TRAIN_REPORT_URL,
    WANDB_EVAL_REPORT_URL,
    WANDB_CURRENT_RUNS_REPORT_URL,
)
from auto_llm.tasks.registry import TASKS

# TODO: avoid hard-coded paths
OUTPUT_DIR = "/vol/auto_llm/sft_models/"
CONFIGS_DIR = ".cache"

GPU_PARAMS = get_gpu_params()


def generate_configs(
    model_names: List[str],
    task: str,
    dataset_path: str,
    instruction_template: str,
    input_template: str,
    output_template: str,
):
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    configs_path = f"{CONFIGS_DIR}/{timestamp_str}_configs"

    print("Config", locals())

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
        dataset_path=dataset_path,
        configs_path=configs_path,
        output_path=OUTPUT_DIR,
        instruction_template=instruction_template,
        input_template=input_template,
        output_template=output_template,
    )

    configurator_outputs = configurator.generate()
    return configurator_outputs, configs_path


def update_file_explorer(configs_path: str):
    return gr.update(label=configs_path, root_dir=configs_path, interactive=True, visible=True)


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
    configured_task = TASKS.get(task)
    return configured_task.sample_trainer_run_config.trainer_data_builder_config.instruction_template


def update_input_textbox(task: str):
    configured_task = TASKS.get(task)
    return configured_task.sample_trainer_run_config.trainer_data_builder_config.input_template


def update_output_textbox(task: str):
    configured_task = TASKS.get(task)
    return configured_task.sample_trainer_run_config.trainer_data_builder_config.output_template


def update_models(task: str, dataset: str, hardware_type: str, hardware_count: int):
    configured_task = TASKS.get(task)
    automator = Automator(
        task_type=configured_task.name,
        dataset=dataset,
        hardware_type=hardware_type,
        hardware_count=hardware_count,
    )

    model_results = automator.get_models_df()

    def bold_max(s):
        is_max = s == s.max()
        return ["font-weight: bold" if v else "" for v in is_max]

    result_cols = model_results.columns[2:]
    model_results = model_results.style.highlight_max(color="#7bb37d", axis=0, subset=result_cols)
    model_results.apply(bold_max, subset=result_cols)

    return automator.model_names, model_results


def update_models_dropdown(model_names: List[str]):
    return gr.update(choices=model_names)


def save_file(file_path: str, file_contents: str):
    if not file_path:
        return None

    try:
        with open(file_path, "w+") as f:
            yaml.safe_dump(yaml.safe_load(file_contents), f, indent=4)
        gr.Info(f"Saved file: {file_path}", duration=5)
    except:
        raise gr.Error(f"YAML file is corrupt. Cannot save file: {file_path}")


def execute_configs(job_id_to_attach: int, configurator_outputs: List[ConfiguratorOutput]):
    if job_id_to_attach != -1:
        executor = SequentialConfigExecutor(configurator_outputs=configurator_outputs, job_id_to_attach=job_id_to_attach)
    else:
        executor = ConfigExecutor(configurator_outputs=configurator_outputs)

    try:
        executor.execute()
        gr.Success("Your jobs are submitted!")
    except:
        raise gr.Error("Your jobs can not be submitted!")


def change_tab(id: int):
    return gr.Tabs(selected=id)


def display_status():
    # TODO: update hard-coded values, including username
    # TODO: consider Config Executor without Slurm
    username = "vsudhi"
    cmd = [
        "squeue",
        "-u",
        username,
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    if not data.get("jobs"):
        return None

    jobs = []
    for job in data["jobs"]:
        if not username in job.get("current_working_directory"):
            continue

        if job.get("job_state") == "CANCELLED":
            continue

        if "language_model" in job.get("name"):
            continue

        job_info = {
            "Job ID": job.get("job_id"),
            "Job Name": job.get("name"),
            "State": job.get("job_state"),
            "Nodes": job.get("nodes"),
            "Partition": job.get("partition"),
            "Dependency": job.get("dependency"),
        }
        jobs.append(job_info)

    df = pd.DataFrame(jobs)
    return df


def update_estimates(config_path: str, gpu_name: str, gpu_count: int):
    if not config_path:
        return None, None

    models_meta = get_model_params()

    if "eval" in config_path:
        flops_estimator = InferenceFlopsEstimator(config_path=config_path, models_meta=models_meta)
    elif "train" in config_path:
        # TODO: models_meta is not updated with the requested model
        flops_estimator = TrainerFlopsEstimator(config_path=config_path, models_meta=models_meta)
    else:
        return None, None

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

    runtime = f"{round(runtime, 2)} seconds"
    emission = f"{round(emission, 2)} grams"

    return runtime, emission


def wandb_report(url):
    iframe = """
<iframe id="my-frame" src={url} style="border:none;height:1024px;width:100%"></iframe>
<script>
window.setInterval("reloadIFrame();", 20000);
function reloadIFrame() {
 var frameHolder=document.getElementById("my-frame");
frameHolder.src={url}
}
</script>
    """.replace("{url}", url).strip()

    return gr.HTML(iframe)


with gr.Blocks() as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Configure", id=0):
            with gr.Row(equal_height=True):
                datasets_list = ""
                datasets_prefix = "https://huggingface.co/datasets"
                for dataset in Automator.get_datasets():
                    datasets_list += f"* [``{dataset}``]({datasets_prefix}/{dataset})\n"

                with gr.Column():
                    dataset_path = gr.Textbox(
                        label="Dataset Path",
                        info=f"Path of the dataset. This can either be remote HuggingFace Datasets paths or local paths. Examples:\n{datasets_list}",
                    )

                with gr.Column():
                    task = gr.Dropdown(
                        label="Task Category",
                        choices=list(TASKS.keys()),
                        value=None,  # noqa
                        multiselect=False,
                        allow_custom_value=True,
                        interactive=True,
                        info=f"Category of the task.",
                    )

            with gr.Row(equal_height=True):
                with gr.Column():
                    hardware = gr.Dropdown(
                        label="Hardware",
                        choices=list(GPU_PARAMS.keys()),
                        value=list(GPU_PARAMS.keys())[0],
                        multiselect=False,
                        allow_custom_value=False,
                        info="Select the device configuration.",
                        interactive=True,
                    )
                with gr.Column():
                    num_hardware = (
                        gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=1,
                            label="Number of GPUs",
                            info="Choose between 1 and 8",
                            interactive=True,
                        ),
                    )
            with gr.Group():
                model_names = gr.Dropdown(
                    label="Models",
                    # TODO: model names should be listed based on the configured task and language.
                    # TODO: also, consider the hardware requirements while selecting models
                    choices=None,
                    value=None,  # noqa
                    multiselect=True,
                    allow_custom_value=True,
                    info="Pick models from the list or of your choice. You can also add models from HuggingFace Models. See: [here](https://huggingface.co/models).",
                )

                with gr.Accordion(label="Model Overview", open=False):
                    gr.Markdown("Based on results from [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).")
                    model_results = gr.Dataframe(value=None)

            with gr.Row(equal_height=True):
                with gr.Column():
                    instruction_template = gr.Textbox(
                        label="Instruction Template",
                        lines=10,
                        info=TrainerDataBuilderConfig.model_fields["instruction_template"].description,
                    )
                with gr.Column():
                    input_template = gr.Textbox(
                        label="Input Template",
                        lines=10,
                        info=TrainerDataBuilderConfig.model_fields["input_template"].description,
                    )
                with gr.Column():
                    output_template = gr.Textbox(
                        label="Output Template",
                        lines=10,
                        info=TrainerDataBuilderConfig.model_fields["output_template"].description,
                    )

            submit_btn = gr.Button("✅ Next")

            configurator_outputs = gr.State()

        with gr.TabItem("Validate", id=1):
            validate_desc = """
            You can review and validate the generated configurations. After making necessary changes, you can save each YAML configuration.
            
            For more information regarding the configuration, see the guidelines [here](https://github.com/ag-sc/auto-llm/blob/add-playground/docs/README.md).
            """
            gr.Markdown(validate_desc)
            with gr.Row():
                with gr.Column(scale=2):
                    config_explorer = gr.FileExplorer(label="", glob="*.yaml", file_count="single", visible=False)

                with gr.Column(scale=2):
                    file_viewer = gr.Code(label="", language="yaml", visible=False, lines=30, max_lines=30)
                    save_btn = gr.Button(value="🗂️ Save configuration", visible=False)

                with gr.Column(scale=1):
                    text = """\
                    ### ⏳ **Estimated runtime**
                    Runtime is estimated based on the configured device and the task complexity.
                    """
                    gr.Markdown(text)
                    estimated_runtime = gr.Label(show_label=False)

                    text = """\
                    ### 🌱 **Estimated CO2 emission**
                    CO2 emission is estimated based on the configured device, the task complexity and the region of use (assumed as Germany).
                    """
                    gr.Markdown(text)
                    estimated_emission = gr.Label(show_label=False)
            with gr.Row():
                validate_submit_btn = gr.Button("✅ Next")

        with gr.TabItem("Run", id=2):
            gr.Markdown("Run the configurations by clicking the button.")
            job_id_to_attach = gr.Textbox(label="Job ID To Attach", value=-1)
            run_btn = gr.Button("✅ Run")

            gr.Markdown("You can see the status of the scheduled jobs here.")
            status_text = gr.Dataframe(
                label="Job Status",
                every=1,
                value=display_status,  # noqa
                show_fullscreen_button=True,
                wrap=True,
                # column_widths=["1px", "1px", "1px", "1px", "1px", "1px"],
            )

        with gr.TabItem("Monitor", id=3):
            with gr.Accordion("Current Runs", open=True):
                wandb_report(WANDB_CURRENT_RUNS_REPORT_URL)

            with gr.Accordion("Trainer Runs", open=False):
                wandb_report(WANDB_TRAIN_REPORT_URL)

            with gr.Accordion("Evaluator Runs", open=False):
                wandb_report(WANDB_EVAL_REPORT_URL)

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
    ).then(
        fn=update_models,
        inputs=[task, dataset_path, hardware, num_hardware[0]],
        outputs=[model_names, model_results],
    ).then(
        fn=update_models_dropdown,
        inputs=[],
        outputs=[model_names],
    )

    configs_path = gr.State()

    submit_btn.click(
        fn=generate_configs,
        inputs=[
            model_names,
            task,
            dataset_path,
            instruction_template,
            input_template,
            output_template,
        ],
        outputs=[configurator_outputs, configs_path],
    ).success(fn=update_file_explorer, inputs=[configs_path], outputs=[config_explorer]).then(fn=change_tab, inputs=[gr.State(1)], outputs=[tabs])

    config_explorer.change(fn=view_yaml_file, inputs=[config_explorer], outputs=[file_viewer]).success(
        fn=update_file_name, inputs=[config_explorer], outputs=[file_viewer]
    ).success(fn=update_save_btn, inputs=[], outputs=[save_btn]).success(
        fn=update_estimates,
        inputs=[config_explorer, hardware, num_hardware[0]],
        outputs=[estimated_runtime, estimated_emission],
    )

    validate_submit_btn.click(fn=change_tab, inputs=[gr.State(2)], outputs=[tabs])

    save_btn.click(
        fn=save_file,
        inputs=[config_explorer, file_viewer],
    )

    run_btn.click(fn=execute_configs, inputs=[job_id_to_attach, configurator_outputs])

if __name__ == "__main__":
    demo.launch()
