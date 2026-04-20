from transformers import AutoModel, AutoTokenizer
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





IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

# Set up argument parser
parser = argparse.ArgumentParser(description="internvl")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--model_size', type=str, default=None, required=True, help='Size of internvl')
parser.add_argument('--data_mode', type=str, default=None, required=True, choices=["train", "val"], help='Data mode ("train" or "val").')
parser.add_argument('--vector_save_dir', type=str, required=True, help='Directory to save the output statistics files.')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
read_dir = args.read_dir
image_base_dir = args.image_dir
model_size = args.model_size
data_mode = args.data_mode
vector_save_dir = args.vector_save_dir



device_map = "auto"

model_name = f"OpenGVLab/InternVL2_5-{model_size}B"

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()

device = model.device

# model = AutoModel.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True,
#     load_in_8bit=True).eval()


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def prepare_inputs(tokenizer, pixel_values, question, generation_config):
    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())


    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()


    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    generation_config['eos_token_id'] = eos_token_id

    query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
    query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')

    return input_ids, attention_mask, query, template, query_to_print, generation_config



def prepare_embed(pixel_values=None, input_model=None, input_ids=None):
    with torch.no_grad():
        vit_embeds = input_model.extract_feature(pixel_values)
    input_embeds = input_model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == input_model.img_context_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    return input_embeds


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

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
    image_path = os.path.join(image_base_dir, item['image']) if image_base_dir else io.BytesIO(base64.b64decode(item['image']))
    
    # img_bytes = base64.b64decode(item['image'])
    # image_ori = Image.open(io.BytesIO(img_bytes))
    

    # Get question from conversations
    pre_question = None
    for conv in item['conversations']:
        if conv['from'] == 'human':
            pre_question = conv['value'].replace('\n<image>', '')
            break
    
    
    question = '<image>\n' + pre_question


    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
    generation_config = dict(max_new_tokens=1, do_sample=False, temperature=0)


    input_ids, attention_mask, query, template, query_to_print, updated_gen_config = prepare_inputs(tokenizer, pixel_values, question, generation_config)

    # multi_embeds = prepare_embed(pixel_values=pixel_values, input_model=model, input_ids=input_ids)

    # Set PyTorch to print full tensors without truncation
    # torch.set_printoptions(threshold=float('inf'))
    # print(input_ids)
    # print(attention_mask)
    # print(query_to_print)
    # print(query)

    


    text_inputs_ids = tokenizer([pre_question], padding=False, return_tensors="pt").input_ids.to(model.device)

    # print('text_inputs_ids:', text_inputs_ids)
    # break

    with torch.no_grad():
        text_embeddings = model.language_model.get_input_embeddings()(text_inputs_ids)
    

    with torch.no_grad():
        vit_embeds = model.extract_feature(pixel_values)


    
    assert vit_embeds.shape[-1] == text_embeddings.shape[-1], "Text and image embeddings must have the same dimension"

    vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(text_embeddings.device)


    image_ind_1 = (input_ids[0] == 92544).nonzero(as_tuple=True)[0].item()
    image_ind_2 = (input_ids[0] == 92545).nonzero(as_tuple=True)[0].item()

    
    image_feature_in_inputid_len = image_ind_2- image_ind_1 - 1


    assert vit_embeds.shape[0] == image_feature_in_inputid_len, f"Image feature length mismatch: {vit_embeds.shape[0]} vs {image_feature_in_inputid_len}"


    text_embedding_list.append(text_embeddings[0, :, :].detach().cpu())
    image_embedding_list.append(vit_embeds.detach().cpu())


    count += 1



text_all_embedding = torch.cat(text_embedding_list, dim=0)
image_all_embedding = torch.cat(image_embedding_list, dim=0)

# Print final shapes
print('text_all_embedding shape:', text_all_embedding.shape)
print('image_all_embedding shape:', image_all_embedding.shape)
print('Current model and data mode:', model_name, data_mode)

print("\n" + "="*50)
print("--- Detailed Per-Feature-Dimension Statistics ---")
# Note: the embedding tensors have shape [total_num_tokens, hidden_size]
# We calculate statistics over the token dimension (dim=0).

# Text Embeddings Statistics
text_mean_per_dim = text_all_embedding.mean(dim=0)
text_std_per_dim = text_all_embedding.std(dim=0, unbiased=False)
print(f"\nText Mean Vector (Shape: {text_mean_per_dim.shape})")
print(f"Text STD Vector (Shape:  {text_std_per_dim.shape})")
print(f"  - Mean of Text Mean Vector: {text_mean_per_dim.mean().item():.6f}")
print(f"  - Mean of Text STD Vector:  {text_std_per_dim.mean().item():.6f}")


# Image Embeddings Statistics
image_mean_per_dim = image_all_embedding.mean(dim=0)
image_std_per_dim = image_all_embedding.std(dim=0, unbiased=False)
print(f"\nImage Mean Vector (Shape: {image_mean_per_dim.shape})")
print(f"Image STD Vector (Shape:  {image_std_per_dim.shape})")
print(f"  - Mean of Image Mean Vector: {image_mean_per_dim.mean().item():.6f}")
print(f"  - Mean of Image STD Vector:  {image_std_per_dim.mean().item():.6f}")

#===================================================================
# --- NEW SECTION: SAVING STATISTICS TO FILES ---
#===================================================================
print("\n" + "="*50)
# 1. Define where to save the files
save_dir = vector_save_dir
print(f"Saving statistics to directory: {save_dir}")

# 2. Create a clean model name for the filename (replaces '/' with '_')
clean_model_name = model_name.replace('/', '_')

# 3. Save the text statistics in a dictionary
text_stats_path = os.path.join(save_dir, f"{clean_model_name}_{data_mode}_text_stats.pt")
torch.save({
    'mean': text_mean_per_dim,
    'std': text_std_per_dim
}, text_stats_path)
print(f"-> Saved text statistics to: {text_stats_path}")

# 4. Save the image statistics in a dictionary
image_stats_path = os.path.join(save_dir, f"{clean_model_name}_{data_mode}_image_stats.pt")
torch.save({
    'mean': image_mean_per_dim,
    'std': image_std_per_dim
}, image_stats_path)
print(f"-> Saved image statistics to: {image_stats_path}")
print("="*50)