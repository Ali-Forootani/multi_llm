


from flask import Flask, request, jsonify
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig

from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image
import torch
from load_web_service_config import LoadWebServicesConfig

WEB_SERVICE_CFG = LoadWebServicesConfig()

import requests

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
    return processor, model


processor, model = load_llava(quantized=True)

app = Flask(__name__)


@app.route('/interact_with_llava', methods=['POST'])
def interact_with_llava():
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = request.get_json()
        prompt = data['prompt']
        url = data['image_url']
        max_output_token = data['max_output_token']
        print("==================")
        print(url)
        print(prompt)
        print("==================")
        prompt = f"USER: <image>\n {prompt} \nASSISTANT:"
        try:
            
            image = Image.open(url)
        except:
            image = None
        
        
        
        
        inputs = processor(prompt, image, return_tensors='pt').to(device, torch.float16)

        output = model.generate(**inputs, max_new_tokens = max_output_token, do_sample=False)
        
        
        
       
        
        response = processor.decode(output[0] , skip_special_tokens=True)
        
        
       
        
        
        #print(processor.decode(output[0][2:], skip_special_tokens=True))
        
        
        
        #output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        #print(processor.decode(output[0][1:], skip_special_tokens=True))

        
        
        """
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=max_output_token)
        print(output)
        response = processor.decode(output[0], skip_special_tokens=True)
        print(response)
        """
        
        
        del output
        del inputs
        del image
        torch.cuda.empty_cache()
        return jsonify({'text': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=WEB_SERVICE_CFG.llava_service_port)
