import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import diffusers
from src.pipeline.pipe import Edit3DFromPromptAnd2DImage
import rembg
import numpy as np
from PIL import Image
import sys
sys.path.append("src/zero123/")


def add_background(img):
    bg = np.array([0,0,0])
    norm_data = np.array(img) / 255.0
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    return img

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Create the model
    model: Edit3DFromPromptAnd2DImage = Edit3DFromPromptAnd2DImage(cfg.models)

    image = None
    # Generate an image
    if cfg.misc.load_image_path == "":
        image = model.generate(cfg.misc.prompt, seed=cfg.misc.seed)
        image.save(cfg.output.generate)
    else:
        # open png image
        import PIL
        image = PIL.Image.open(cfg.misc.load_image_path).convert('RGB')
        image.save(cfg.output.generate)

    kwargs = {
        "edit_kwargs": {
            "guidance_scale": cfg.models.hyperparams.instruct.guidance_scale,
            "image_guidance_scale": cfg.models.hyperparams.instruct.image_guidance_scale,
        },
    }

    model(
        input_image=image,
        edit_prompt=cfg.misc.edit,
        save_path="./out/"
    )



if __name__ == "__main__":
    my_app()