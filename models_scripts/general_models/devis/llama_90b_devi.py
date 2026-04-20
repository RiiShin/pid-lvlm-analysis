from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from PIL import Image
import requests
import copy
import torch
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import math
import numpy as np


import sys
import warnings
import json
import os
import time


import argparse

import base64
import io


# Set up argument parser
parser = argparse.ArgumentParser(description="llama")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--model_size', type=str, default=None, required=True, help='Size of llama')
parser.add_argument('--data_mode', type=str, default=None, required=True, choices=["train", "val"], help='Data mode ("train" or "val").')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
read_dir = args.read_dir
image_base_dir = args.image_dir
model_size = args.model_size
data_mode = args.data_mode

model_name = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct"



model = MllamaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)



tokenizer = AutoTokenizer.from_pretrained(model_name)


print('Tokenizer:', tokenizer)



# Read the input JSON file
input_json_path = read_dir + data_mode + ".json"  # 2000 samples now!!!!


# image_base_dir = "/home/xiu.lixin/msra/data/MMVP_Images" 
warnings.filterwarnings("ignore")


device = "cuda"
device_map = "auto"


print('------------------')
print(model_name)
print('------------------')
# Load the dataset
with open(input_json_path, 'r') as f:
    data = json.load(f)


text_embedding_list = []
image_embedding_list = []


count = 0

# Process each example
for item in tqdm(data):
    # Load image
    # image_path = os.path.join(image_base_dir, item['image'])

    if image_base_dir:
        image_path = os.path.join(image_base_dir, item['image'])
        image_ori = Image.open(image_path).convert("RGB")
    else:
        img_bytes = base64.b64decode(item['image'])
        image_ori = Image.open(io.BytesIO(img_bytes))
    

    # Get question from conversations
    question = None
    for conv in item['conversations']:
        if conv['from'] == 'human':
            question = conv['value'].replace('\n<image>', '')
            break
    
    
    message_dual = [
        # {"role": "system", "content": "You are a helpful assistant. Answer with exactly “Yes” or “No” (no other words).",},
    {   
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }
    ]

    input_text = processor.apply_chat_template(message_dual, add_generation_prompt=True)

    # print('input_text:', input_text)


    inputs = processor(
        image_ori,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values'],
            aspect_ratio_ids=inputs['aspect_ratio_ids'],
            aspect_ratio_mask=inputs['aspect_ratio_mask'],
        )
        cross_attention_states = vision_outputs[0]
        
        cross_attention_states = model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.hidden_size
        )

    
        inputs_embeds_check = model.language_model.model.embed_tokens(inputs['input_ids'])

    text_ind_1 = (inputs['input_ids'][0] == model.config.image_token_index).nonzero(as_tuple=True)[0].item()
    text_ind_2 = (inputs['input_ids'][0] == 128009).nonzero(as_tuple=True)[0].item()

    
    text_feature = inputs_embeds_check[0, text_ind_1+1:text_ind_2, :]


    text_embedding_list.append(text_feature.detach().cpu())
    image_embedding_list.append(cross_attention_states.view(-1,model.hidden_size).detach().cpu())


    count += 1

    if count < 10:  # Print only the first 10 examples
        print('text_feature:', text_feature.shape)
        print('cross_attention_states:', cross_attention_states.shape)



text_all_embedding = torch.cat(text_embedding_list, dim=0)
image_all_embedding = torch.cat(image_embedding_list, dim=0)

print('text_all_embedding:', text_all_embedding.shape)
print('image_all_embedding:', image_all_embedding.shape)

print('Current model and data mode:', model_name, data_mode)

# # Calculate standard deviation for text embeddings
# # Flatten the batch and sequence dimensions to calculate std per feature dimension
# text_embeddings_flat = text_all_embedding.view(-1, text_all_embedding.shape[-1])
# text_std = torch.std(text_embeddings_flat, dim=0)
# print('Text embedding standard deviation shape:', text_std.shape)
# # Optionally, print the mean standard deviation across all feature dimensions
# print('Mean text embedding standard deviation:', torch.mean(text_std).item())


# # Calculate standard deviation for image embeddings
# # Flatten the batch and patch dimensions to calculate std per feature dimension
# image_embeddings_flat = image_all_embedding.view(-1, image_all_embedding.shape[-1])
# image_std = torch.std(image_embeddings_flat, dim=0)
# print('Image embedding standard deviation shape:', image_std.shape)
# # Optionally, print the mean standard deviation across all feature dimensions
# print('Mean image embedding standard deviation:', torch.mean(image_std).item())


print(text_all_embedding.flatten().std(unbiased=False))
print(image_all_embedding.flatten().std(unbiased=False))

# You can save these standard deviations if needed
# torch.save(text_std, f'text_std_{data_mode}.pt')
# torch.save(image_std, f'image_std_{data_mode}.pt')
