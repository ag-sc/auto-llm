import gradio as gr

with gr.Blocks() as demo:
    text = """
    # **AutoLLM** - train and evaluate your Language Models effortlessly! 😊
    ...
    
    """

    gr.Markdown(text)

    gr.Image(
        "auto_llm/ui/images/uni-bielefeld-logo.png",
        show_label=False,
        container=False,
        height=100,
        width=100,
        show_download_button=False,
        show_fullscreen_button=False,
    )

if __name__ == "__main__":
    demo.launch()
