from typing import Any
import hydra
import torch
from typing import Optional, Union
from PIL import Image
import os
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class Edit3DFromPromptAnd2DImage():

    def __init__(self, cfg, dtype=None):
        self.cfg = cfg
        if dtype is None:
            dtype = torch.float16
        self.sd = hydra.utils.call(cfg.sd, torch_dtype=dtype, cache_dir=self.cfg.cache_dir).to(cfg.device)
        self.inp = hydra.utils.call(cfg.inpainting, torch_dtype=dtype, cache_dir=self.cfg.cache_dir).to(cfg.device)
        self.instruct = hydra.utils.call(cfg.instruct, torch_dtype=dtype, cache_dir=self.cfg.cache_dir).to(cfg.device)
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
        
    
    def __call__(self, sd_prompt: str = "", 
                input_image: Optional[Union[torch.Tensor, Image.Image]] = None,
                edit_prompt: str = "",
                inpaint_prompt: str = "",
                inpaint_mask: Optional[Union[torch.Tensor, Image.Image]] = None,
                save_path: str = "",
                **kwargs: Any) -> Any:
        if sd_prompt and input_image:
            raise ValueError("Cannot provide both sd_prompt and input_image")
        if sd_prompt:
            image = self.generate(sd_prompt, kwargs=kwargs)
            image.save(os.path.join(save_path, "sd.png"))
        elif input_image:
            image = input_image        
        if edit_prompt:
            image = self.edit(edit_prompt, image, kwargs=kwargs)
            image.save(os.path.join(save_path, "edit.png"))
        if inpaint_mask:
            if not inpaint_prompt:
                inpaint_prompt = edit_prompt
            image = self.inpaint(inpaint_prompt, image, inpaint_mask, kwargs=kwargs)
            image.save(os.path.join(save_path, "inpaint.png"))
        
        novel_views = self.novelViews(image, kwargs=kwargs)
        novel_views.save(os.path.join(save_path, "novel_views.png"))
        return novel_views