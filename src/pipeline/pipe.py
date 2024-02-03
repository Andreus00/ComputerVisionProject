from typing import Any
import hydra
import torch
from typing import Optional, Union
from PIL import Image
import os
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import rembg
import torchvision.transforms.functional as TF
from ..zero123.ldm.util import instantiate_from_config
from ..zero123 import ldm
from ..zero123.ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange

import math

class Edit3DFromPromptAnd2DImage():

    def __init__(self, cfg, dtype=None):
        self.cfg = cfg
        if dtype is None:
            dtype = torch.float16
        self.sd = hydra.utils.call(cfg.sd, torch_dtype=dtype, cache_dir=self.cfg.cache_dir)
        self.inp = hydra.utils.call(cfg.inpainting, torch_dtype=dtype, cache_dir=self.cfg.cache_dir)
        self.instruct = hydra.utils.call(cfg.instruct, torch_dtype=dtype, cache_dir=self.cfg.cache_dir)
        self.instruct.scheduler = EulerAncestralDiscreteScheduler.from_config(self.instruct.scheduler.config)
        self.zero_plus = hydra.utils.call(cfg.zero_plus, torch_dtype=dtype)
        self.zero = instantiate_from_config(cfg.zero.model)
        old_state = torch.load("cache/zero123-xl.ckpt", map_location="cpu")["state_dict"]
        self.zero.load_state_dict(old_state)

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def generate(self, prompt, seed=None, kwargs={}):
        self.sd.to(self.cfg.device)
        if seed:
            self.seed_everything(seed)
        res = self.sd(prompt, **kwargs).images[0]
        self.sd.to("cpu")
        return res
    
    @torch.no_grad()
    def edit(self, prompt, image, seed=None, kwargs={}):
        self.instruct.to(self.cfg.device)
        if seed:
            self.seed_everything(seed)
        res = self.instruct(prompt, 
                             image=image, 
                             **kwargs).images[0]
        self.instruct.to("cpu")
        return res
    
    @torch.no_grad()
    def inpaint(self, prompt, image, mask_image, seed=None, kwargs={}):
        self.inp.to(self.cfg.device)
        if seed:
            self.seed_everything(seed)
        res = self.inp(prompt, 
                        image=image,
                        mask_image=mask_image, 
                        **kwargs).images[0]
        self.inp.to("cpu")
        return res
    
    @torch.no_grad()
    def novelViewsZeroPlus(self, cond, seed=None, kwargs={}):
        '''
        Run zero_123 plus on an image
        '''
        self.zero_plus.to(self.cfg.device)
        if seed:
            self.seed_everything(seed)
        res = self.zero_plus(cond, **kwargs).images[0]
        self.zero_plus.to("cpu")
        # image_novel = rembg.remove(res)
        # background = Image.new("RGB", image_novel.size, (0, 0, 0))
        # background.paste(image_novel, mask = image_novel.split()[3])
        return res

    @torch.no_grad()
    def unpack_zero_plus_out(self, zero_plus_out):
        '''
        Unpack the output of zero_plus into 6 images
        '''
        images = []
        sub_images_size = 320
        for i in range(6):
            left  = (i % 2)  * sub_images_size
            right = left     + sub_images_size - 1 
            upper = (i // 2) * sub_images_size
            lower = upper + sub_images_size - 1
            
            images.append(zero_plus_out.crop(box = (left, upper, right, lower)))

        return images
    
    @torch.no_grad()
    def callZero(self, input_im, sampler, x, y, z, model, n_samples=1, w=256, h=256, ddim_steps=50, ddim_eta=1.0, scale=1.0):
        with model.ema_scope():
            if isinstance(input_im, Image.Image):
                input_im = TF.to_tensor(input_im).unsqueeze(0).to(self.cfg.device)
            input_im = input_im.float()

            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device).to(c.dtype)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim), min=0, max=1.0).cpu()
            output_ims = []
            for x_sample in x_samples_ddim:
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
            return output_ims


    @torch.no_grad()
    def novelViewsZero(self, zero_plus_images, seed=None, kwargs={}):
        '''
        Take each one of the 6 images from zero_plus and run zero_123 on them.
        For each image generate 4 novel views with respectively -30, +30 degrees rotation for azimuth and -30 +20 elevation.
        '''
        self.zero.to(self.cfg.device)
        if seed:
            self.seed_everything(seed)
        
        azimuths = [-20, 20]
        elevations = [-10, +10]
        novel_views = []
        for image in zero_plus_images[:2]:
            sampler = DDIMSampler(self.zero)
            image = TF.resize(image, (256, 256), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            for azimuth in azimuths:
                new_view = self.callZero(input_im = image, 
                                         sampler = sampler, 
                                         x = 0, 
                                         y = azimuth, 
                                         z = 0, 
                                         model = self.zero,
                                         )
                novel_views.append(new_view)#, guidance_scale = 0.0, elevation=torch.as_tensor([0]).to(self.cfg.device), azimuth=torch.as_tensor([azimuth]).to(self.cfg.device), distance=torch.as_tensor([3]).to(self.cfg.device), **kwargs).images[0])
            for elevation in elevations:
                new_view = self.callZero(input_im = image, 
                                         sampler = sampler, 
                                         x = elevation, 
                                         y = 0, 
                                         z = 0, 
                                         model = self.zero,
                                         )
                novel_views.append(new_view)

        self.zero.to("cpu")
        return novel_views
    
        
    @torch.no_grad()
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
            edit_kwargs = {}
            if "edit_kwargs" in kwargs:
                edit_kwargs = kwargs["edit_kwargs"]
            image = self.edit(edit_prompt, image, kwargs=edit_kwargs)
            image.save(os.path.join(save_path, "edit.png"))
        if inpaint_mask:
            if not inpaint_prompt:
                inpaint_prompt = edit_prompt
            image = self.inpaint(inpaint_prompt, image, inpaint_mask, kwargs=kwargs)
            image.save(os.path.join(save_path, "inpaint.png"))
        
        novel_views = self.novelViewsZeroPlus(image, kwargs=kwargs)
        novel_views.save(os.path.join(save_path, "novel_views.png"))

        novel_views = self.unpack_zero_plus_out(novel_views)

        novel_views = self.novelViewsZero(novel_views, kwargs=kwargs)
        for i, images in enumerate(novel_views):
            for j, image in enumerate(images):
                image.save(os.path.join(save_path, f"novel_view_{i}_{j}.png"))
        return novel_views
    


