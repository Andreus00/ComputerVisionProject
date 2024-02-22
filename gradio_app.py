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
    return model.generate(prompt).resize((512, 512), Image.Resampling.BICUBIC)
    

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
    # rescale image
    image = image.resize((512, 512), Image.Resampling.BICUBIC)
    kwargs = {
        "guidance_scale": guidance_scale,
        "image_guidance_scale": image_guidance_scale,
    }
    return model.edit(prompt, image, kwargs=kwargs).resize((512, 512), Image.Resampling.BICUBIC)


def zerp_plus(image):
    image = Image.fromarray(image).resize((512, 512), Image.Resampling.BICUBIC)
    novel_views = model.novelViewsZeroPlus(image)
    novel_unpacked = model.unpack_zero_plus_out(novel_views)
    return novel_unpacked

# def zero(images):
#     print(f"Images: {images}")
#     imgs = []
#     for img_metadata in images:
#         response = requests.get(img_metadata["data"])
#         img = Image.open(BytesIO(response.content))
#         imgs.append(img)
#     novel_views = model.novelViewsZero(imgs)
#     return novel_views


def save_imgs(edited_img, zero_plus_img, save_name):
    save_name = save_name.replace(" ", "_").replace(".", "").replace("/", "")
    if not os.path.exists(f"images/{save_name}"):
        os.makedirs(f"images/{save_name}")
    edited_img = Image.fromarray(edited_img).resize((512, 512), Image.Resampling.BICUBIC)
    edited_img.save(f"images/{save_name}/edit.png")
    for i, img_metadata in enumerate(zero_plus_img):
        response = requests.get(img_metadata["data"])
        img = Image.open(BytesIO(response.content))
        img.save(f"images/{save_name}/novel_zero_plus_view_{i}.png")
    # for i, img_metadata in enumerate(zero_gallery):
    #     response = requests.get(img_metadata["data"])
    #     img = Image.open(BytesIO(response.content))
    #     img.save(f"images/{save_name}/novel_view_{i}.png")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def start_gradio(conf: DictConfig) -> None:
    global cfg, model
    cfg = conf
    model = Edit3DFromPromptAnd2DImage(cfg.models)
    with gr.Blocks() as demo:

        # Stable Diffusion
            
        name = gr.Textbox(label="Stable Diffusion Prompt")
        output_sd = gr.Image(label="Generated Image", source="upload")
        greet_btn = gr.Button("generate SD image")
        greet_btn.click(fn=generate_sd, inputs=name, outputs=output_sd, api_name="generate_sd")

        # Upload
        uploaded = gr.Image(label="Uploaded Image", source="upload")
        upload_btn = gr.UploadButton("Upload Image")


        # Edit
        guidance_scale = gr.Slider(minimum=0, maximum=20, value=5.5, label="Guidance Scale")
        image_guidance_scale = gr.Slider(minimum=0, maximum=20, value=1.5, label="Image Guidance Scale")
        edit_prompt = gr.Textbox(label="Edit Prompt", value="make it made of gold")
        edit_btn = gr.Button("Edit Image")
        edited_img = gr.Image(label="Edited Image", source="upload")
        edit_btn.click(fn=edit_image, 
                        inputs=[uploaded, output_sd, edit_prompt, guidance_scale, image_guidance_scale],
                        outputs=edited_img, 
                        api_name="edit_image")

        # Zero Plus
        zero_plus_btn = gr.Button("Zero Plus")
        zero_plus_img = gr.Gallery(label="Zero Plus Images")
        zero_plus_img.style(grid=6)
        zero_plus_btn.click(fn=zerp_plus, inputs=edited_img, outputs=zero_plus_img, api_name="zero_plus")

        # # Zero
        # zero_btn = gr.Button("Zero")
        # zero_gallery = gr.Gallery(label="Zero Images")
        # zero_gallery.style(columns=4, rows=6)
        # zero_btn.click(fn=zero, inputs=zero_plus_img, outputs=zero_gallery, api_name="zero")

        # save images
        save_name = gr.Textbox(label="Save Name")
        save_btn = gr.Button("Save Image")
        save_btn.click(fn=save_imgs, inputs=[edited_img, zero_plus_img, save_name], api_name="save")


    demo.launch(share=True)

if __name__ == "__main__":
    start_gradio()