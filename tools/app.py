import logging
import os
# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt")
import gradio as gr
from dotenv import load_dotenv

from configs.config import Config

from infer.modules.vc.modules import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model =='':
        return "You need to name your model. For example: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./assets/weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."
            
def show_available(filepath):
    return os.listdir(filepath)
  


with gr.Blocks(theme=gr.themes.Soft(), title="RVC-DEMO-Web 💻") as app:
    gr.HTML("<h1> RVC DEMO 💻 </h1>")
            )
            sid = gr.Dropdown(label=("inference"), choices=sorted(names))
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=("speaker id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])
            gr.Markdown(
                value=("12key is recommended for converting male to female, and -12key is recommended for converting female to male. If the sound range explodes and the tone is distorted, you can adjust it to the appropriate range yourself.")
            )
            vc_input3 = gr.Audio(label="Upload audio (less than 90 seconds in length)")
            vc_transform0 = gr.Number(label=("Transposition (integer, number of semitones, octave up 12 octave down -12)"), value=0)
            f0method0 = gr.Radio(
                label=("Choose the pitch extraction algorithm. You can use pm to speed up the input singing. Harvest has good bass but is extremely slow. Crepe has good effect but consumes GPU."),
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="pm",
                interactive=True,
            )
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label=(">=3, use median filtering on the harvest pitch recognition result, the value is the filter radius, which can weaken the mute sound."),
                value=3,
                step=1,
                interactive=True,
            )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=("Feature retrieval library file path, if empty, use the drop-down selection result"),
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label=("Automatically detect index path, drop-down selection (dropdown)"),
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=("Search feature proportion"),
                value=0.88,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=("Post-processing resampling to the final sampling rate, 0 means no resampling"),
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=("The input source volume envelope replaces the output volume envelope fusion ratio. The closer it is to 1, the more output envelope is used. "),
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=("Protect unvoiced consonants and breathing sounds to prevent artifacts such as electronic sound tearing. Do not turn it on when it reaches 0.5. Turn it down to increase the protection but may reduce the indexing effect."),
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(label=("F0 curve file, optional, one pitch per line, replacing the default F0 and rising and falling tones"))
            but0 = gr.Button("Convert"), variant="primary")
            vc_output1 = gr.Textbox(label=("Output information"))
            vc_output2 = gr.Audio(label=("Output audio (three dots in the lower right corner, click to download)"))

with gr.Row():
        with gr.Tabs():
            with gr.TabItem("2.Choose an audio file:"):
                audio_picker = gr.Dropdown(label="",choices=show_available('audios'),value='',interactive=True)
            with gr.TabItem("(Or upload a new file here)"):
                dropbox = gr.File(label="Drop an audio here. (You can also drop a .pth or .index file here)")
                dropbox.upload(fn=upload_file, inputs=[dropbox],outputs=[audio_picker,model_picker,dropbox])
        audio_refresher = gr.Button("Refresh")
        audio_refresher.click(fn=refresh,inputs=[],outputs=[audio_picker,model_picker])


            but0.click(
                vc.vc_single,
                [
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    # file_big_npy1,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
               )


app.launch(share=True)
