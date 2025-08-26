import json

import gradio as gr
import requests

from lms.service import LmsRestDelegate

SAMPLE_MODEL_CONFIG = {
    "gemma-2-2b": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "device": "cuda:0",
        "max_context_length": 1024,
    },
    "gemma-2-2b-sft": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "adapter": "/vol/auto_llm/sft_models/pico_ad_gemma-2-2b_non-conv_zero-shot",
        "device": "cuda:0",
        "max_context_length": 1024,
    },
}

SAMPLE_GENERATION_CONFIG = {"max_new_tokens": 100, "do_sample": True, "top_p": 1}


def connect_to_lms(lms_port: str):
    lms_endpoint = f"http://localhost:{lms_port}"

    try:
        response = requests.get(f"{lms_endpoint}/docs")
        if response.status_code == 200:
            lms_api = LmsRestDelegate(endpoint=lms_endpoint)
            status = "🟢 Connected to LMS!"
        else:
            raise gr.Error(f"Could not connect to LMS API: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        raise gr.Error(f"Could not connect to LMS API: {e}")

    return lms_api, status


def get_available_models(lms_api: LmsRestDelegate):
    models = lms_api.available_models()
    return models


def load_models(lms_api: LmsRestDelegate, cfg: str):
    cfg = json.loads(cfg)
    try:
        lms_api.load_models(config=cfg)

    except:
        raise gr.Error(f"Could not load model: {cfg}")

    status = "🟢 Models loaded!"

    return status


def generate(
    lms_api: LmsRestDelegate, model_name: str, text: str, generation_kwargs: str
):
    generation_kwargs = json.loads(generation_kwargs)
    response = lms_api.generate(
        model_name=model_name, texts=[text], generation_kwargs=generation_kwargs
    )[0]
    return response


def update_models_dropdown(models):
    return gr.update(
        choices=models, interactive=True, value="", allow_custom_value=False
    )


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            connect_status = gr.Textbox(
                label="Connection Status:",
                value="🔴 LMS not connected!",
                interactive=False,
                container=False,
            )
            with gr.Accordion("Connect", open=False):
                gr.Markdown(
                    "Connect to the **Language Model Service** endpoint. This should be running in `localhost`."
                )
                port = gr.Textbox(label="Port", value="9985", interactive=True)
                connect_btn = gr.Button("Connect")

                lms_api = gr.State()

        with gr.Column(scale=1):
            load_status = gr.Textbox(
                label="Models Status:",
                value="🔴 Models not loaded!",
                interactive=False,
                container=False,
            )
            with gr.Accordion("Load", open=False):
                gr.Markdown("Load the models here.")
                cfg = gr.Code(
                    label="Model Configuration",
                    value=json.dumps(SAMPLE_MODEL_CONFIG, indent=4),
                    language="json",
                    interactive=True,
                )
                load_btn = gr.Button("Load")
                loaded_models = gr.State()

    connect_btn.click(
        fn=connect_to_lms,
        inputs=[port],
        outputs=[lms_api, connect_status],
    )

    with gr.Accordion("Explore", open=True):
        model_name = gr.Dropdown(
            label="Model",
            choices=[],
            interactive=False,
            value="Please connect to LMS and load the models to use them!",
            allow_custom_value=True,
        )

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input", value="", lines=10, interactive=True)
            with gr.Column():
                generation_kwargs = gr.Code(
                    label="Generation kwargs",
                    value=json.dumps(SAMPLE_GENERATION_CONFIG, indent=4),
                    language="json",
                    interactive=True,
                    lines=15,
                )

        generate_btn = gr.Button("Generate")
        # parse_as_markdown = gr.Checkbox(label="Parse as JSON", value=True)

        response = gr.Code(
            label="Response",
            value="",
            container=True,
            lines=10,
        )

        generate_btn.click(
            fn=generate,
            inputs=[lms_api, model_name, text, generation_kwargs],
            outputs=[response],
        )

        load_btn.click(
            fn=load_models, inputs=[lms_api, cfg], outputs=[load_status]
        ).then(fn=get_available_models, inputs=[lms_api], outputs=[loaded_models]).then(
            fn=update_models_dropdown,
            inputs=[loaded_models],
            outputs=[model_name],
        )

if __name__ == "__main__":
    demo.launch()
