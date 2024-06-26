# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:03:19 2024

@author: forootan
"""

from flask import Flask, request, jsonify
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig

from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image
import torch
from load_web_service_config import LoadWebServicesConfig

from transformers import AutoTokenizer

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




def load_llava(quantized=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = setting_directory(4) + "\\Llavar_repo\\LLaVA\\llava-1.5-7b-hf"
    
   
    if quantized and str(1.6) in model_path:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("===============================================")
        print("Loading the quantized version of the model:")
        print("===============================================")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quantization_config, device_map="auto")
        processor = LlavaNextProcessor.from_pretrained(
            model_path)
    else:
        print("Loading the full model:")
        
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
            ).to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        
        #model = LlavaNextForConditionalGeneration.from_pretrained(
        #    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    return processor, model, tokenizer


processor, model, tokenizer = load_llava(quantized=True)




#app = Flask(__name__)

import requests



prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

#image_file = "C:\\Users\\forootan\\AppData\\Local\\Temp\\28\\gradio\\2c25f7701b6944d6cf976a320a08eff66f12a9b9\\flat-tire.jpg"

raw_image = Image.open(requests.get(image_file, stream=True).raw)

#raw_image = Image.open(image_file)



inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][1:], skip_special_tokens=True))


print(output)

response = processor.decode(output[0][1:], skip_special_tokens=True)
print(response)
  
response_2 = processor.decode(output[0], skip_special_tokens=True)
print(response_2)
  
response_3 = processor.decode(output[0][2:], skip_special_tokens=True)
print(response_3)
