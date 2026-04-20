from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
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
from conversation import get_conv_template

import sys
import warnings
import json
import os
import time


import argparse

import base64
import io


# Set up argument parser
parser = argparse.ArgumentParser(description="fuyu")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--data_mode', type=str, default=None, required=True, choices=["train", "val"], help='Data mode ("train" or "val").')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
read_dir = args.read_dir
image_base_dir = args.image_dir
data_mode = args.data_mode

device = "cuda"
device_map = "auto"


model_name = "adept/fuyu-8b"


# model = AutoModel.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True,
#     load_in_4bit=True).eval()   # load_in_8bit=True for 8-bit quantization, if needed


model = FuyuForCausalLM.from_pretrained(model_name, device_map=device_map)



processor = FuyuProcessor.from_pretrained(model_name)

# print('Processor:', processor)


tokenizer = AutoTokenizer.from_pretrained(model_name)


print('Tokenizer:', tokenizer)



# Read the input JSON file
input_json_path = read_dir + data_mode + ".json"  # 2000 samples now!!!!


# image_base_dir = "/home/xiu.lixin/msra/data/MMVP_Images" 
warnings.filterwarnings("ignore")


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


    print('mode:', image_ori.mode)
    
    
    # Get question from conversations
    question = None
    for conv in item['conversations']:
        if conv['from'] == 'human':
            question = conv['value'].replace('\n<image>', '\n')
            break
    
    inputs = processor(text=question, images=image_ori, return_tensors="pt").to(device)


    inputs_embeds_overall = model.language_model.get_input_embeddings()(inputs['input_ids'])

    with torch.no_grad():
        image_ori_embeddings = model.vision_embed_tokens(inputs['image_patches'][0].to(model.vision_embed_tokens.weight.dtype)).squeeze(0).to(inputs_embeds_overall.device)


    # print('image_ori_embeddings shape:', image_ori_embeddings.shape)


    # vision_embeddings = image_ori_embeddings.mean(dim=0, keepdim=True)
    # vision_embeddings = vision_embeddings[0]



    text_ind_1 = (inputs['input_ids'][0] == 1).nonzero(as_tuple=True)[0].item()
    text_ind_2 = (inputs['input_ids'][0] == 71375).nonzero(as_tuple=True)[0][-1].item()

    
    text_feature = inputs_embeds_overall[0, text_ind_1+1:text_ind_2, :]


    # text_feature = text_feature.mean(dim=0, keepdim=True)  # Shape: [1, 896]


    text_embedding_list.append(text_feature.detach().cpu())
    image_embedding_list.append(image_ori_embeddings.detach().cpu())


    count += 1

    if count < 10:  # Print only the first 10 examples
        print('text_feature:', text_feature.shape)
        print('cross_attention_states:', image_ori_embeddings.shape)



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
