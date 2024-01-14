import gradio as gr

def greet(name):
    return "Hello " + name + "!!"
  
with gr.Blocks(title="🔊",theme=gr.themes.Base(primary_hue="rose",neutral_hue="zinc")) as app:
    with gr.Row():
        gr.HTML("<img  src='file/a.png' alt='image'>")

app.launch()