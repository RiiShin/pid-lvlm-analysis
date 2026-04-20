from transformers import AutoTokenizer, AutoModel

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



torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print(f"Random seed set to {42} for reproducibility.")



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'



# Set up argument parser
parser = argparse.ArgumentParser(description="Run InternVL3 model inference with specified parameters.")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--save_dir', type=str, default=None, required=True, help='Directory to save the output JSON files.')
parser.add_argument('--num_choices', type=int, default=4, help='Number of choices for the model to predict (default: 4).')
parser.add_argument('--model_type', type=str, default=None, required=True)
parser.add_argument('--model_size', type=str, default=None, required=True)
parser.add_argument('--data_mode', type=str, default=None, required=True, choices=["train", "val"], help='Data mode ("train" or "val").')
parser.add_argument('--devi_vector_dir', type=str, default=None, required=True, help='Directory containing the pre-calculated deviation vectors.')


# Parse arguments
args = parser.parse_args()


# Assign parsed arguments to variables
read_dir = args.read_dir
image_base_dir = args.image_dir
save_dir = args.save_dir
num_choices = args.num_choices
model_size = args.model_size
data_mode = args.data_mode
devi_vector_dir = args.devi_vector_dir


# Construct model name based on size
model_name = f"OpenGVLab/InternVL3-{model_size}B" if args.model_type == '3' else f"OpenGVLab/InternVL2_5-{model_size}B"
# Construct save file name (moved after argument parsing)
save_json_name = f"/internvl3_{model_size}.json" if args.model_type == '3' else f"/internvl2_5_{model_size}.json"


print(f"Using model: {model_name}")
print(f"Data mode: {data_mode}")
print(f"Save JSON name: {save_json_name}")


scale_factor = 1.0



tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

print('Tokenizer:', tokenizer)

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map='auto').eval()



# Read the input JSON file
input_json_path = read_dir + data_mode + ".json"  # 2000 samples now!!!!


output_json_path = save_dir + "/" + data_mode + save_json_name   # check if the path is for 2000 samples!!!!


# image_base_dir = "/home/xiu.lixin/msra/data/MMVP_Images" 
warnings.filterwarnings("ignore")


device = model.device


# ==============================================================================
# --- (NEW) 1. LOAD PRE-CALCULATED EMBEDDING STATISTICS ---
# ==============================================================================
print("\n" + "="*50)
print("Loading pre-calculated embedding statistics...")

# IMPORTANT: Make sure these paths are correct!
# You will likely have one set of stats for 'train' and one for 'val'.
# Make sure you load the stats that correspond to the `data_mode` you are running.
stats_dir = devi_vector_dir
clean_model_name = model_name.replace('/', '_')

text_stats_path = os.path.join(stats_dir, f"{clean_model_name}_train_text_stats.pt")
image_stats_path = os.path.join(stats_dir, f"{clean_model_name}_train_image_stats.pt")

# Load the dictionaries
try:
    text_stats = torch.load(text_stats_path)
    image_stats = torch.load(image_stats_path)
except FileNotFoundError as e:
    print(f"ERROR: Could not find statistics files. Make sure these paths are correct:\n- {text_stats_path}\n- {image_stats_path}")
    raise e

# Extract the tensors and move them to the correct GPU/CPU device
text_mean_vector = text_stats['mean'].to(device)
text_std_vector = text_stats['std'].to(device)

image_mean_vector = image_stats['mean'].to(device)
image_std_vector = image_stats['std'].to(device)

print("Successfully loaded embedding statistics.")
print(f"Text stats loaded from: {text_stats_path}")
print(f"Image stats loaded from: {image_stats_path}")
print("="*50 + "\n")


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


# Get token IDs for 'A' and 'B'
# We assume 'A' and 'B' are represented by single tokens.
ids_a = tokenizer.encode('A', add_special_tokens=False)
if not (len(ids_a) == 1 and isinstance(ids_a[0], int)):
    raise ValueError(f"Token 'A' is not represented by a single token ID. Got: {ids_a}")
token_a_id = ids_a[0]

ids_b = tokenizer.encode('B', add_special_tokens=False)
if not (len(ids_b) == 1 and isinstance(ids_b[0], int)):
    raise ValueError(f"Token 'B' is not represented by a single token ID. Got: {ids_b}")
token_b_id = ids_b[0]

if num_choices == 4:
    ids_c = tokenizer.encode('C', add_special_tokens=False)
    if not (len(ids_c) == 1 and isinstance(ids_c[0], int)):
        raise ValueError(f"Token 'C' is not represented by a single token ID. Got: {ids_c}")
    token_c_id = ids_c[0]

    ids_d = tokenizer.encode('D', add_special_tokens=False)
    if not (len(ids_d) == 1 and isinstance(ids_d[0], int)):
        raise ValueError(f"Token 'D' is not represented by a single token ID. Got: {ids_d}")
    token_d_id = ids_d[0]


print(f"Token ID for 'A': {token_a_id}")
print(f"Token ID for 'B': {token_b_id}")
if num_choices == 4:
    print(f"Token ID for 'C': {token_c_id}")
    print(f"Token ID for 'D': {token_d_id}")




all_choice_token_ids = [token_a_id, token_b_id, token_c_id, token_d_id] if num_choices == 4 else [token_a_id, token_b_id]


confidence_threshold = 0.3


def get_choice_distributions(logits, num_options):
    """
    Analyzes model logits to extract probability distributions for multiple-choice questions.

    This function calculates two types of probability distributions:
    1.  Forced-Choice Probs: A distribution over valid options that sums to 1 (relative confidence).
    2.  Absolute Probs: The raw model confidence for each potential choice from the full vocabulary.

    Args:
        logits (torch.Tensor): The raw output logits from the model. 
                               Assumed shape: [batch_size, seq_len, vocab_size].
        all_choice_token_ids (list[int]): A list containing the token IDs for all possible
                                          choices, e.g., [id_A, id_B, id_C, id_D]. The
                                          function assumes a fixed maximum number of choices.
        num_options (int): The number of valid choices for this specific question (e.g., 2, 3, or 4).

    Returns:
        dict: A dictionary containing:
            - 'forced_choice_probs': A padded tensor with relative probabilities over valid choices.
            - 'absolute_probs': A tensor with the absolute probability of each potential choice.
            - 'overall_predicted_token_id': The token ID of the model's top-1 prediction from the entire vocabulary.
    """

    last_token_logits = logits[0, -1, :]
    
    overall_predicted_token_id = torch.argmax(last_token_logits).item()


    full_vocab_probs = torch.softmax(last_token_logits, dim=-1)
    absolute_probs = full_vocab_probs[all_choice_token_ids]

    max_options = len(all_choice_token_ids)
    forced_choice_probs = torch.zeros(max_options)
 

    valid_token_ids = all_choice_token_ids[:num_options]
    valid_logits = last_token_logits[valid_token_ids]
    

    valid_choice_probs = torch.softmax(valid_logits, dim=-1)
    
    forced_choice_probs[:num_options] = valid_choice_probs


    choice_confidence_score = torch.sum(absolute_probs[:num_options])
    
    # If the confidence is below the threshold, override with a uniform distribution
    if choice_confidence_score < confidence_threshold:
        if num_options > 0:
            uniform_prob = 1.0 / num_options
            forced_choice_probs[:num_options] = uniform_prob
            # The rest of the tensor is already zero, which is correct.

    return forced_choice_probs, absolute_probs, [overall_predicted_token_id]


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



# print('Model config: ', model.config)

print(f"Tokenizer vocab size: {len(tokenizer)}")


eos_token_id = tokenizer.eos_token_id
print(f'Padding token id: {tokenizer.pad_token_id}')
print(f'EOS token id: {tokenizer.eos_token_id}')
# print(f'EOS token: {tokenizer.eos_token}')


# print(f'Decode three token ids: {tokenizer.decode([151665, 151666, 151667], skip_special_tokens=False)}')


print('------------------')
print(model_name)
print('output_json_path:', output_json_path)
print('------------------')
# Load the dataset
with open(input_json_path, 'r') as f:
    data = json.load(f)





count = 0

right_answer = 0


# Process each example
for item in tqdm(data):
    # Load image
    # image_path = os.path.join(image_base_dir, item['image'])

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


    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)
    generation_config = dict(max_new_tokens=1, do_sample=False, temperature=0)


    input_ids, attention_mask, query, template, query_to_print, updated_gen_config = prepare_inputs(tokenizer, pixel_values, question, generation_config)


    with torch.no_grad():
        multi_embeds = prepare_embed(pixel_values=pixel_values, input_model=model, input_ids=input_ids)



    text_ind_1 = (input_ids[0] == 151666).nonzero(as_tuple=True)[0].item()
    text_ind_2 = (input_ids[0] == 151645).nonzero(as_tuple=True)[0][1].item()

    image_ind_1 = (input_ids[0] == 151665).nonzero(as_tuple=True)[0].item()
    image_ind_2 = (input_ids[0] == 151666).nonzero(as_tuple=True)[0].item()

    text_feature = multi_embeds[0, text_ind_1+1:text_ind_2, :]
    image_feature = multi_embeds[0, image_ind_1+1:image_ind_2, :]

    # print('text_feature shape:', text_feature.shape)
    # print('image_feature shape:', image_feature.shape)

    text_feature = text_feature.mean(dim=0, keepdim=True)  # Shape: [1, 896]
    image_feature = image_feature.mean(dim=0, keepdim=True)  # Shape: [1, 896]


    text_feature = text_feature[0]
    image_feature = image_feature[0]

    inputs_embeds_check = multi_embeds.clone()

    
    inputs_image_masked_dict = inputs_embeds_check.clone()

    image_slice = inputs_image_masked_dict[:, image_ind_1+1:image_ind_2, :].clone()

    # image_noise = torch.randn_like(image_slice, device=image_slice.device, dtype=image_slice.dtype) * 3 * DEV_DICT[model_name][0]

    # Create replacement for the visual embeddings using the GLOBAL stats
    visual_replacement_embeds = torch.randn_like(image_slice) * (image_std_vector * scale_factor) + image_mean_vector
    

    inputs_image_masked_dict[:, image_ind_1+1:image_ind_2, :] = visual_replacement_embeds.clone()

    


    inputs_text_masked_dict = inputs_embeds_check.clone()

    text_slice = inputs_text_masked_dict[:, text_ind_1+1:text_ind_2, :].clone()

    # text_noise = torch.randn_like(text_slice, device=text_slice.device, dtype=text_slice.dtype) * 3 * DEV_DICT[model_name][1]

    
    lang_replacement_embeds = torch.randn_like(text_slice) * (text_std_vector * scale_factor) + text_mean_vector


    inputs_text_masked_dict[:, text_ind_1+1:text_ind_2, :] = lang_replacement_embeds.clone()



    # with torch.no_grad():
        # multi_output = model.language_model.generate(
        #         inputs_embeds=multi_embeds,
        #         attention_mask=attention_mask,
        #         use_cache=True,
        #         return_dict_in_generate=True,  # 
        #         output_scores=True,   #
        #         **updated_gen_config,
        #     )
    
    


    with torch.no_grad():

        cont = model.language_model.generate(
                inputs_embeds=multi_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict_in_generate=True,  # 
                output_scores=True,   #
                **updated_gen_config,
            )
        

        multi_logits = torch.stack(cont.scores, dim=1)
        # print('multi_logits shape:', multi_logits.shape)
        multi_probs, multi_orig_probs, output_tokens = get_choice_distributions(multi_logits, item["num_options"])


        assert int(cont.sequences[0][-1]) == int(output_tokens[0])


        output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



        image_single_output = model.language_model.generate(
                                inputs_embeds=inputs_text_masked_dict,
                                attention_mask=attention_mask,
                                use_cache=True,
                                return_dict_in_generate=True,  # 
                                output_scores=True,   #
                                **updated_gen_config,
                            )

        # Get logits and hidden states
        logits_image_single= torch.stack(image_single_output.scores, dim=1)
        # print('visual_logits shape:', visual_logits.shape)

        
        visual_probs, visual_orig_probs, visual_tokens = get_choice_distributions(logits_image_single, item["num_options"])




        text_single_output = model.language_model.generate(
                                inputs_embeds=inputs_image_masked_dict,
                                attention_mask=attention_mask,
                                use_cache=True,
                                return_dict_in_generate=True,  # 
                                output_scores=True,   #
                                **updated_gen_config,
                            )

        # Get logits and hidden states
        lang_logits = torch.stack(text_single_output.scores, dim=1)
        

        
        language_two_probs, language_two_orig_probs, lang_tokens = get_choice_distributions(lang_logits, item["num_options"])



        
    
    response = output_text

    # Update the GPT response in the data
    for conv in item['conversations']:
        if conv['from'] == 'gpt':
            conv['label'] = conv['value']
            conv['value'] = response
            conv['v_feature'] = image_feature.to(dtype=torch.float32, device='cpu').detach().numpy().tolist()
            conv['l_feature'] = text_feature.to(dtype=torch.float32, device='cpu').detach().numpy().tolist()
            conv['v_prob'] = visual_probs.to('cpu').detach().numpy().tolist()
            conv['l_prob'] = language_two_probs.to('cpu').detach().numpy().tolist()
            conv['vl_prob'] = multi_probs.to('cpu').detach().numpy().tolist()
            conv['v_orig_prob'] = visual_orig_probs.to('cpu').detach().numpy().tolist()
            conv['l_orig_prob'] = language_two_orig_probs.to('cpu').detach().numpy().tolist()
            conv['vl_orig_prob'] = multi_orig_probs.to('cpu').detach().numpy().tolist()


            if count % 100 == 0:
                print('------------------')
                for key in ['label', 'value', 'v_prob', 'l_prob', 'vl_prob', 
                            'v_orig_prob', 'l_orig_prob', 'vl_orig_prob']:
                    print(f"{key}: {conv[key]}")
                for key in ['v_feature', 'l_feature']:
                    print(f"{key}: {len(conv[key])}")
                print('------------------')
            break
    
    # Normalize and compare the response with the label
    normalized_response = response.lower().strip()
    normalized_label = conv['label'].lower().strip()

    if normalized_response == normalized_label:
        right_answer += 1

    count += 1


accuracy = right_answer / count if count > 0 else 0
print(f"Processed {count} samples. Current accuracy: {accuracy:.4f}")


# Save the results
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=2)