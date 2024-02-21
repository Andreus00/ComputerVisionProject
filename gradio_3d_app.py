import gradio as gr
import hydra
from omegaconf import DictConfig
from PIL import Image
import logging
import os
import sys
sys.path.append("src/zero123/")

from train_2 import main

logger = logging.getLogger(__name__)

cfg = None
base_path = "images"

def load_from_path(save_name):
    save_name = save_name.replace(" ", "_").replace(".", "").replace("/", "")
    if not os.path.exists(os.path.join(base_path, save_name)):
        raise ValueError(f"Path {save_name} does not exist")
    edited_img = Image.open(f"images/{save_name}/edit.png")
    zero_plus_img = []
    for i in range(6):
        zero_plus_img.append(Image.open(f"images/{save_name}/novel_zero_plus_view_{i}.png"))
    zero_gallery = []
    # for i in range(6 * 4):
    #     zero_gallery.append(Image.open(f"images/{save_name}/novel_view_{i}.png"))
    
    return edited_img, zero_plus_img
    

def load_imgs(save_name):
    return load_from_path(save_name)

def generate_gs(path):
    main(os.path.join(base_path, path))

    return f"images/{path}/gifs/1000.gif", f"images/{path}/gifs/500.gif", f"images/{path}/gifs/300.gif"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def start_gradio(conf: DictConfig) -> None:
    global cfg, model
    cfg = conf
    with gr.Blocks() as demo:

        # Load images
        name = gr.Textbox(label="Saved image name")
        frontal_img = gr.Image(label="Frontal Image", source="upload")
        zero_plus_side_imgs = gr.Gallery(label="Zero Plus Side Image")
        zero_plus_side_imgs.style(grid=6)
        # zero_side_imgs = gr.Gallery(label="Zero Side Image")
        # zero_side_imgs.style(columns=4, rows=6)
        greet_btn = gr.Button("load")
        greet_btn.click(fn=load_imgs, inputs=name, outputs=[frontal_img, zero_plus_side_imgs], api_name="load_images")

        # Start Gaussian Splatting
        start_btn = gr.Button("Start Gaussian Splatting")
        gaussian_splatting_gallery = gr.Gallery(label="Gaussian Splatting Result")
        gaussian_splatting_gallery.style(grid=3)
        start_btn.click(fn=generate_gs, 
                        inputs=[name], 
                        outputs=gaussian_splatting_gallery, 
                        api_name="start_gaussian_splatting")

    demo.launch()

if __name__ == "__main__":
    start_gradio()