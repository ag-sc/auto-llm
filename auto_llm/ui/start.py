import gradio as gr

from auto_llm.ui.pages import home, automate, train, eval, playground

head_html = """
<style>
    .title-container {
        position: absolute;
        top: 0;
        left: 0;
        padding: 10px;
        font-size: 2em;
        font-weight: bold;
    }
</style>
<div class="title-container">
    AutoLLM
</div
"""


with gr.Blocks(
    title="AutoLLM", css_paths="auto_llm/ui/css/app.css", head=head_html
) as demo:
    home.demo.render()

# add other pages
with demo.route(name="📁 Automate", path="/automate"):
    automate.demo.render()
with demo.route(name="⚙️ Train", path="/train"):
    train.demo.render()
with demo.route(name="⚖️ Evaluate", path="/evaluate"):
    eval.demo.render()
with demo.route(name="▶️ Playground", path="/playground"):
    playground.demo.render()

if __name__ == "__main__":
    demo.launch()
