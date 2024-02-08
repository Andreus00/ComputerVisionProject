import gradio as gr
import hydra
from omegaconf import DictConfig, OmegaConf
from src.pipeline.pipe import Edit3DFromPromptAnd2DImage
from PIL import Image
import logging
import requests
from io import BytesIO
import os
import sys
sys.path.append("src/zero123/")

logger = logging.getLogger(__name__)

cfg = None
model = None
# first_image = None
# output_sd = None
# uploaded = None

def generate_sd(prompt: str):
    global cfg, model
    return model.generate(prompt).resize((256, 256), Image.Resampling.BICUBIC)
    

def edit_image(image_upl, image_sd, prompt, guidance_scale, image_guidance_scale):
    global cfg, model
    logger.info(f"Editing {image_upl} with prompt: {prompt}")
    if image_upl is not None:
        image = Image.fromarray(image_upl)
    elif image_sd is not None:
        image = Image.fromarray(image_sd)
    else:
        raise ValueError("No image provided")
    
    print(f"Image: {image}")
    kwargs = {
        "guidance_scale": guidance_scale,
        "image_guidance_scale": image_guidance_scale,
    }
    return model.edit(prompt, image, kwargs=kwargs).resize((256, 256), Image.Resampling.BICUBIC)


def zerp_plus(image):
    image = Image.fromarray(image).resize((256, 256), Image.Resampling.BICUBIC)
    novel_views = model.novelViewsZeroPlus(image)
    novel_unpacked = model.unpack_zero_plus_out(novel_views)
    return novel_unpacked

def zero(images):
    print(f"Images: {images}")
    imgs = []
    for img_metadata in images:
        response = requests.get(img_metadata["data"])
        img = Image.open(BytesIO(response.content))
        imgs.append(img)
    novel_views = model.novelViewsZero(imgs)
    return novel_views


def save_imgs(edited_img, zero_plus_img, zero_gallery, save_name):
    save_name = save_name.replace(" ", "_").replace(".", "").replace("/", "")
    if not os.path.exists(f"images/{save_name}"):
        os.makedirs(f"images/{save_name}")
    edited_img.save(f"images/{save_name}/edit.png")
    for i, img_metadata in enumerate(zero_plus_img):
        response = requests.get(img_metadata["data"])
        img = Image.open(BytesIO(response.content))
        img.save(f"images/{save_name}/novel_zero_plus_view_{i}.png")
    for i, img_metadata in enumerate(zero_gallery):
        response = requests.get(img_metadata["data"])
        img = Image.open(BytesIO(response.content))
        img.save(f"images/{save_name}/novel_view_{i}.png")

def load_imgs(save_name):
    save_name = save_name.replace(" ", "_").replace(".", "").replace("/", "")
    if not os.path.exists(f"images/{save_name}"):
        raise ValueError(f"Image {save_name} does not exist")
    edited_img = Image.open(f"images/{save_name}/edit.png")
    zero_plus_img = []
    for i in range(6):
        zero_plus_img.append(Image.open(f"images/{save_name}/novel_zero_plus_view_{i}.png"))
    zero_gallery = []
    for i in range(6 * 4):
        zero_gallery.append(Image.open(f"images/{save_name}/novel_view_{i}.png"))
    
    return edited_img, zero_plus_img, zero_gallery

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
        zero_side_imgs = gr.Gallery(label="Zero Side Image")
        zero_side_imgs.style(columns=4, rows=6)
        greet_btn = gr.Button("load")
        greet_btn.click(fn=load_imgs, inputs=name, outputs=[frontal_img, zero_plus_side_imgs, zero_side_imgs], api_name="load_images")

        # Start Gaussian Splatting
        start_btn = gr.Button("Start Gaussian Splatting")
        gaussian_splatting_gallery = gr.Gallery(label="Gaussian Splatting")
        start_btn.click(fn=generate_gs, inputs=[frontal_img, zero_plus_side_imgs, zero_side_imgs], outputs=frontal_img, api_name="start_gaussian_splatting")

    demo.launch()

if __name__ == "__main__":
    start_gradio()