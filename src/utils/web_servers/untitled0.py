# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:02:55 2024

@author: forootan
"""

from diffusers import DiffusionPipeline



import sys
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


def load_stable_diffusion_xl_base_1_0():
    """
    Load the Stable Diffusion XL Base 1.0 model.

    Returns:
        StableDiffusionXLPipeline: The loaded model pipeline.
    """
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    
    model_path = setting_directory(4) + "\\Llavar_repo\\LLaVA\\stable-diffusion-xl-base-1.0"
    
    model_path_2 = setting_directory(4) + "\\Llavar_repo\\LLaVA\\SDXL-Lightning"
    
    
    #pipeline = DiffusionPipeline.from_pretrained(model_path)
    
    #base = "stabilityai/stable-diffusion-xl-base-1.0"
    #repo = "ByteDance/SDXL-Lightning"
    # Use the correct ckpt for your step setting!
    ckpt = "sdxl_lightning_4step_lora.safetensors"

    # Load model.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    
    pipe.load_lora_weights(hf_hub_download(model_path_2, ckpt))
    pipe.fuse_lora()

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


#pipe = load_stable_diffusion_xl_base_1_0()


from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

model_path = setting_directory(4) + "\\Llavar_repo\\LLaVA\\stable-diffusion-xl-base-1.0"

model_path_2 = setting_directory(4) + "\\Llavar_repo\\LLaVA\\SDXL-Lightning" 
ckpt = "sdxl_lightning_4step_lora.safetensors"

lora_weights = model_path_2 + ckpt


# Load model.
pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")


pipe.load_lora_weights(model_path_2,
                       weight_name= "sdxl_lightning_4step_lora.safetensors")



pipe.fuse_lora()








