import gradio as gr
import os, shutil

import subprocess, os
assets_folder = "assets"
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)
files = {
    "rmvpe/rmvpe.pt":"https://huggingface.co/Rejekts/project/resolve/main/rmvpe.pt",
    "hubert/hubert_base.pt":"https://huggingface.co/Rejekts/project/resolve/main/hubert_base.pt",
    "pretrained_v2/D40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/D40k.pth",
    "pretrained_v2/G40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/G40k.pth",
    "pretrained_v2/f0D40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/f0D40k.pth",
    "pretrained_v2/f0G40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/f0G40k.pth"
}
for file, link in files.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', link, '-O', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
            
def show_available(filepath):
    return os.listdir(filepath)
  
def upload_file(file):
    audio_formats = ['.wav', '.mp3', '.ogg', '.flac', '.aac']
    file_name, file_extension = os.path.splitext(file.name)
    if file_extension.lower() in audio_formats:
        shutil.move(file.name,'audios')
    elif file_extension.lower().endswith('.pth'):
        shutil.move(file.name,'assets/weights')
    elif file_extension.lower().endswith('.index'):
        shutil.move(file.name,'logs')
    else:
        print("Filetype not compatible")
    return {"choices":show_available('audios'),"__type__": "update"}

with gr.Blocks() as app:
    with gr.Row():
        dropbox = gr.Dropbox(label="Upload files")
        audio_picker = gr.Dropdown(label="",choices=show_available('audios'))
        dropbox.upload(fn=upload_file, inputs=['dropbox'],outputs=['audio_picker'])

app.launch()  