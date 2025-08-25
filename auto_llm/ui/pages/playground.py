import json

import gradio as gr
import requests
from lms.interface.generator_interface import GeneratorRestDelegate

MOCK_CONFIG = {
    "gemma-2-2b": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "model_type": "AutoModelForCausalLM",
        "tokenizer": "google/gemma-2-2b",
        "use_fast": True,
        "change_pad_token": True,
        "device": "cuda:0",
        "use_accelerate": False,
    },
    "gemma-2-2b-sft": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "model_type": "AutoModelForCausalLM",
        "tokenizer": "google/gemma-2-2b",
        "adapter": "/vol/auto_llm/sft_models/pico_ad_gemma-2-2b_non-conv_zero-shot",
        "use_fast": True,
        "change_pad_token": True,
        "device": "cuda:0",
        "use_accelerate": False,
    },
}


def connect_to_lms(lms_port: str):
    lms_endpoint = f"http://localhost:{lms_port}/"

    try:
        response = requests.get(f"{lms_endpoint}docs")
        if response.status_code == 200:
            lms_api = GeneratorRestDelegate(endpoint=lms_endpoint)
            status = "🟢 Connected!"
        else:
            raise gr.Error(f"Could not connect to LMS API: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        raise gr.Error(f"Could not connect to LMS API: {e}")

    return lms_api, status


def get_available_models(lms_api: GeneratorRestDelegate):
    models = lms_api.available_models()


def load_models(lms_api: GeneratorRestDelegate, cfg: str):
    cfg = json.loads(cfg)
    try:
        lms_api.load_model(config=cfg)

    except:
        raise gr.Error(f"Could not load model: {cfg}")

    status = "🟢 Loaded!"

    return status, loaded_models


def generate(
    lms_api: GeneratorRestDelegate,
    model_name: str,
    text: str,
    max_new_tokens: str,
    split_lines: bool,
    temperature: str,
    frequency_penalty: str,
    presence_penalty: str,
):
    generate_kwargs = {
        "model_name": model_name,
        "max_new_tokens": int(max_new_tokens),
        "split_lines": split_lines,
        "temperature": float(temperature),
        "frequency_penalty": float(frequency_penalty),
        "presence_penalty": float(presence_penalty),
    }

    response = lms_api.generate(texts=[text], **generate_kwargs)[0]
    return response


def update_models_dropdown(model_name_dd, models):
    model_name_dd = gr.Dropdown(label="Model", choices=models)
    return model_name_dd


with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown(
            "Connect to the **Language Model Service** endpoint. This should be running in `localhost`."
        )
        port = gr.Textbox(label="Port", value="9985", interactive=True)
        connect_btn = gr.Button("Connect")
        status = gr.Textbox(
            label="Status", value="🔴 Not Connected!", interactive=False
        )
        lms_api = gr.State()

    connect_btn.click(
        fn=connect_to_lms,
        inputs=[port],
        outputs=[lms_api, status],
    )

    with gr.Accordion("Load", open=False):
        gr.Markdown("Load the models here.")
        cfg = gr.Code(
            label="Model Configuration",
            value=json.dumps(MOCK_CONFIG, indent=4),
            language="json",
            interactive=True,
        )
        load_btn = gr.Button("Load")
        loaded_models = gr.State()

        load_status = gr.Textbox(
            label="Status", value="🔴 Not Loaded!", interactive=False
        )

    with gr.Accordion("Explore", open=True):
        model_name = gr.Textbox(label="Model")

        max_new_tokens = gr.Textbox(
            label="Max New Tokens", value="10", interactive=True
        )
        split_lines = gr.Checkbox(label="Split Lines", value=True)
        temperature = gr.Textbox(label="Temperature", value="0", interactive=True)
        frequency_penalty = gr.Textbox(
            label="Frequency Penalty", value="2", interactive=True
        )
        presence_penalty = gr.Textbox(
            label="Presence Penalty", value="2", interactive=True
        )

        text = gr.Textbox(label="Input", value="")

        generate_btn = gr.Button("Generate")
        # parse_as_markdown = gr.Checkbox(label="Parse as JSON", value=True)

        response = gr.Markdown(label="Response", value="", container=True)

        generate_btn.click(
            fn=generate,
            inputs=[
                lms_api,
                model_name,
                text,
                max_new_tokens,
                split_lines,
                temperature,
                frequency_penalty,
                presence_penalty,
            ],
            outputs=[response],
        )

        load_btn.click(
            fn=load_models, inputs=[lms_api, cfg], outputs=[load_status, loaded_models]
        )

if __name__ == "__main__":
    demo.launch()
