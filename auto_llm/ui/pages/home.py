import gradio as gr

text = """<h1 align="center">
AutoLLM
</h1>

<p align="center">
    <strong>⚙️ Train and ⚖️ Evaluate your LMs effortlessly!</strong>
</p>

<p align="center">
| <a href="https://llm4kmu.de/"><b>Website</b></a> 
| <a href="https://wandb.ai/llm4kmu/projects"><b>Reports</b></a> 
| <a href="https://www.linkedin.com/company/llm4kmu/"><b>LinkedIn</b></a>
| <a href="https://github.com/ag-sc/auto-llm"><b>Code</b></a>
|
</p>

---

**AutoLLM** supports you in finding the **right** open source model, architecture and training method for your application. Inspired by "Auto-ML" methods, **AutoLLM** automatically determines the optimal LLM configuration for a problem, train and evaluate different LLMs for your application. You can choose from different open-source models, training techniques and evaluation metrics.

The platform is part of the project "LLM4KMU". 


> Optimierter Einsatz von Open Source Large Language Models (LLMs) in kleinen und mittelständischen Unternehmen (KMUs). Mit Mitteln der Europäischen Union gefördert. 
> 
> **#efre #efrenrw #EUinmyRegion**

"""

with gr.Blocks() as demo:
    gr.Markdown(text)

    with gr.Row():
        with gr.Column():
            gr.Image(
                value="auto_llm/ui/images/logo.png",
                show_label=False,
                container=False,
                height=200,
                # width=150,
                show_download_button=False,
                show_fullscreen_button=False,
            )
        with gr.Column(scale=1):
            ...

if __name__ == "__main__":
    demo.launch(width="15%")
