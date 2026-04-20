from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import requests
import copy
import torch
from tqdm import tqdm

import sys
import warnings
import json
import os
import time


import argparse

import base64
import io


# Set up argument parser
parser = argparse.ArgumentParser(description="qwen")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--model_size', type=str, default=None, required=True, help='Size of qwen')
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


model_name = f"Qwen/Qwen2-VL-{model_size}B-Instruct"

# Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
min_pixels = 256*28*28
max_pixels = 1280*28*28

processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

# print('Processor:', processor)


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
        # {"role": "system", "content": "You are a helpful assistant.",},
    {   
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_ori,
            },
            {"type": "text", "text": question},
        ],
    }
    ]

    # print('question:', question)


    text = processor.apply_chat_template(
    message_dual, tokenize=False, add_generation_prompt=True
    )

    # print('text:', text)


    image_inputs, video_inputs = process_vision_info(message_dual)
    # print('image_inputs:', image_inputs)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # print('inputs:', inputs)



    # Getting visual_features!!!!!
    pixel_values = inputs['pixel_values'].type(model.visual.dtype)

    with torch.no_grad():
        vision_embeddings_orig = model.visual(pixel_values, grid_thw=inputs['image_grid_thw'])

    # vision_embeddings = vision_embeddings_orig.mean(dim=0, keepdim=True)  
    # vision_embeddings = vision_embeddings[0]

    # print('vision_embeddings:', vision_embeddings_orig.shape)

    
    text_inputs_ids = tokenizer([question], padding=False, return_tensors="pt").input_ids.to(model.device)


    with torch.no_grad():
        text_embeddings = model.model.embed_tokens(text_inputs_ids)

    
    # print('text_embeddings:', text_embeddings.shape)


    text_embedding_list.append(text_embeddings[0, :, :].detach().cpu())
    image_embedding_list.append(vision_embeddings_orig.detach().cpu())


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