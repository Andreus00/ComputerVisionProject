import hydra
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class Edit2DFromPrompt():

    def __init__(self, cfg, dtype=None):
        self.cfg = cfg
        if dtype is None:
            dtype = torch.float16
        self.sd = hydra.utils.call(cfg.sd, torch_dtype=dtype).to(cfg.device)
        self.inp = hydra.utils.call(cfg.inpainting, torch_dtype=dtype).to(cfg.device)
        self.instruct = hydra.utils.call(cfg.instruct, torch_dtype=dtype).to(cfg.device)
        self.instruct.scheduler = EulerAncestralDiscreteScheduler.from_config(self.instruct.scheduler.config)
        self.zero = hydra.utils.call(cfg.zero, torch_dtype=dtype).to(cfg.device)
        

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def generate(self, prompt, seed=None, kwargs={}):
        if seed:
            self.seed_everything(seed)
        return self.sd(prompt, **kwargs).images[0]
    
    def edit(self, prompt, image, seed=None, kwargs={}):
        if seed:
            self.seed_everything(seed)
        return self.instruct(prompt, 
                             image=image, 
                             **kwargs).images[0]
    
    def inpaint(self, prompt, image, mask_image, seed=None, kwargs={}):
        if seed:
            self.seed_everything(seed)
        return self.inp(prompt, 
                        image=image,
                        mask_image=mask_image, 
                        **kwargs).images[0]
    
    def novelViews(self, cond, seed=None, kwargs={}):
        if seed:
            self.seed_everything(seed)
        return self.zero(cond, **kwargs).images[0]