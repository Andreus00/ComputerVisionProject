import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import diffusers


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main():
    # load the zero-edited image and create a loader for gaussian splatting that
    # uses the zero-edited image as a reference.
    pass