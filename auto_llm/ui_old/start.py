import gradio as gr

from auto_llm.ui_old.pages import home, automate, train, eval, playground, define

with gr.Blocks(
    title="AutoLLM",
    css_paths="auto_llm/ui/css/app.css",
) as demo:
    home.demo.render()

# add other pages
with demo.route(name="📚 Define", path="/define"):
    define.demo.render()
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
