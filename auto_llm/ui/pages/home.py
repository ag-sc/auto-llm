import gradio as gr

with gr.Blocks() as demo:
    text = """
    # **AutoLLM** - ⚙️ Train and ⚖️ Evaluate your Language Models effortlessly! 😊
    
    **AutoLLM** supports you in finding the **right** open source model, architecture and training method for your application. 
    
    Inspired by "Auto-ML" methods, **AutoLLM** automatically determines the optimal LLM configuration for a problem, train and evaluate different LLMs for your application. 
    
    You can choose from different open-source models, training techniques and evaluation metrics.
    
    **Code:** [here](https://github.com/ag-sc/auto-llm)
    
    ## 📢 Announcements
    
    ✅ Now supports ``SftTrainer`` with `conversational` and `non-conversational` datasets. Read more [here](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support).
    
    ✅ Now supports all benchmarks in `lm-eval-harness`. Read more [here](https://github.com/EleutherAI/lm-evaluation-harness).
    
    ✅ ...
    
    """

    gr.Markdown(text)

    with gr.Row():
        with gr.Column():
            gr.Image(
                "auto_llm/ui/images/uni-bielefeld-logo.png",
                show_label=False,
                container=False,
                # height=100,
                width=150,
                show_download_button=False,
                show_fullscreen_button=False,
            )
        with gr.Column(scale=2):
            ...

if __name__ == "__main__":
    demo.launch(width="25%")
