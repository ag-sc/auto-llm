import gradio as gr

from auto_llm.ui.pages import home, data, train, eval

with gr.Blocks() as demo:
    home.demo.render()

# add other pages
with demo.route(name="📁 Data", path="/data"):
    data.demo.render()
with demo.route(name="⚙️ Train", path="/train"):
    train.demo.render()
with demo.route(name="⚖️ Evaluate", path="/evaluate"):
    eval.demo.render()

if __name__ == "__main__":
    demo.launch()
