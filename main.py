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
    if cfg.misc.load_image_path == "":
        image = model.generate(cfg.misc.prompt, seed=cfg.misc.seed)
        image.save(cfg.output.generate)
    else:
        # open png image
        import PIL
        image = PIL.Image.open(cfg.misc.load_image_path).convert('RGB')
        image.save(cfg.output.generate)


    # Edit an image
    edit_kwargs = {
        "guidance_scale": cfg.models.hyperparams.instruct.guidance_scale,
        "image_guidance_scale": cfg.models.hyperparams.instruct.image_guidance_scale,
    }
    edit_image = model.edit(cfg.misc.edit, image, seed=cfg.misc.seed, kwargs=edit_kwargs)
    edit_image.save(cfg.output.edit)

    # zero123
    edit_image_novel = model.novelViews(edit_image, seed=cfg.misc.seed)
    edit_image_novel.save(cfg.output.zero_edited)

    # zero123
    image_novel = model.novelViews(image, seed=cfg.misc.seed)
    image_novel.save(cfg.output.zero)



if __name__ == "__main__":
    my_app()