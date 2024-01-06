import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import diffusers
from src.pipeline.pipe import Edit2DFromPrompt


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Create the model
    model = Edit2DFromPrompt(cfg.models)

    # Generate an image
    image = model.generate(cfg.misc.prompt, seed=cfg.misc.seed)
    image.save(cfg.output.generate)

    # Edit an image
    edit_kwargs = {
        "guidance_scale": 10,
        "image_guidance_scale": 1.5,
    }
    image = model.edit(cfg.misc.edit, image, seed=cfg.misc.seed, kwargs=edit_kwargs)
    image.save(cfg.output.edit)

    # zero123
    image = model.novelViews(image, seed=cfg.misc.seed)
    image.save(cfg.output.zero)



if __name__ == "__main__":
    my_app()