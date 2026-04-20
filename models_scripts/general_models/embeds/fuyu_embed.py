# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
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



torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print(f"Random seed set to {42} for reproducibility.")




##### Check add or replace carefully!!!!

# Set up argument parser
parser = argparse.ArgumentParser(description="Run Fuyu model inference with specified parameters.")
parser.add_argument('--read_dir', type=str, default=None, required=True, help='Directory containing the input JSON files.')
parser.add_argument('--image_dir', type=str, default='', help='Image directory (if applicable). None if images are embedded in JSON.')
parser.add_argument('--save_dir', type=str, default=None, required=True, help='Directory to save the output JSON files.')
parser.add_argument('--num_choices', type=int, default=4, help='Number of choices for the model to predict (default: 4).')
parser.add_argument('--data_mode', type=str, default=None, required=True, choices=["train", "val"], help='Data mode ("train" or "val").')
parser.add_argument('--devi_vector_dir', type=str, default=None, required=True, help='Directory containing the pre-calculated deviation vectors.')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
# mask_type = args.mask_type
mask_type = 'use'
read_dir = args.read_dir
image_base_dir = args.image_dir
save_dir = args.save_dir
num_choices = args.num_choices
data_mode = args.data_mode
devi_vector_dir = args.devi_vector_dir


# Construct model name based on size
model_name = "adept/fuyu-8b"
# Construct save file name (moved after argument parsing)
save_json_name = f"/fuyu_{mask_type}.json"


print(f"Using model: {model_name}")
print(f"Mask type: {mask_type}")
print(f"Data mode: {data_mode}")
print(f"Save JSON name: {save_json_name}")


scale_factor = 1.0


model = FuyuForCausalLM.from_pretrained(model_name, device_map="cuda:0")


processor = FuyuProcessor.from_pretrained(model_name)

# print('Processor:', processor)


tokenizer = AutoTokenizer.from_pretrained(model_name)


print('Tokenizer:', tokenizer)



# Read the input JSON file
input_json_path = read_dir + data_mode + ".json"  # 2000 samples now!!!!


output_json_path = save_dir + "/" + data_mode + save_json_name   # check if the path is for 2000 samples!!!!


# image_base_dir = "/home/xiu.lixin/msra/data/MMVP_Images" 
warnings.filterwarnings("ignore")


device = "cuda"
device_map = "auto"


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



# Get token IDs for 'A' and 'B'
# We assume 'A' and 'B' are represented by single tokens.
ids_a = tokenizer.encode('a', add_special_tokens=False)
token_a_id = ids_a[1]

ids_b = tokenizer.encode('b', add_special_tokens=False)
token_b_id = ids_b[1]

if num_choices == 4:
    ids_c = tokenizer.encode('c', add_special_tokens=False)
    token_c_id = ids_c[1]

    ids_d = tokenizer.encode('d', add_special_tokens=False)
    token_d_id = ids_d[1]



assert len(ids_a) == len(ids_b) == 2, "Expected two tokens for 'A' and 'B'."
if num_choices == 4:
    assert len(ids_c) == len(ids_d) == 2, "Expected two tokens for 'C' and 'D'."


print(f"Token ID for 'a': {token_a_id}")
print(f"Token ID for 'b': {token_b_id}")
if num_choices == 4:
    print(f"Token ID for 'c': {token_c_id}")
    print(f"Token ID for 'd': {token_d_id}")




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




print('Model config: ', model.config)





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
            question = conv['value'].replace('\n<image>', '\n')
            question = question.replace('\nA.', '\na.')
            question = question.replace('\nB.', '\nb.')
            question = question.replace('\nC.', '\nc.')
            question = question.replace('\nD.', '\nd.')
            break
    
    inputs = processor(text=question, images=image_ori, return_tensors="pt").to(device)

    # print('input_ids:', inputs['input_ids'])


    # print('returned input texts:', tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False)[0])


    inputs_embeds_overall = model.language_model.get_input_embeddings()(inputs['input_ids'])

    with torch.no_grad():
        image_ori_embeddings = model.vision_embed_tokens(inputs['image_patches'][0].to(model.vision_embed_tokens.weight.dtype)).squeeze(0).to(inputs_embeds_overall.device)


    # print('image_ori_embeddings shape:', image_ori_embeddings.shape)


    vision_embeddings = image_ori_embeddings.mean(dim=0, keepdim=True)
    vision_embeddings = vision_embeddings[0]



    text_ind_1 = (inputs['input_ids'][0] == 1).nonzero(as_tuple=True)[0].item()
    text_ind_2 = (inputs['input_ids'][0] == 71375).nonzero(as_tuple=True)[0][-1].item()


    text_feature = inputs_embeds_overall[0, text_ind_1+1:text_ind_2, :]


    text_feature = text_feature.mean(dim=0, keepdim=True)  # Shape: [1, 896]


    text_feature = text_feature[0]



    # image_noise = torch.randn_like(image_ori_embeddings, device=image_ori_embeddings.device, dtype=image_ori_embeddings.dtype) * 3 * DEV_DICT[model_name][0]


    # Create replacement for the visual embeddings using the GLOBAL stats
    visual_replacement_embeds = torch.randn_like(image_ori_embeddings) * (image_std_vector * scale_factor) + image_mean_vector
    
        

    image_masked_inputs_embeds = model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds_overall,
        continuous_embeddings=[visual_replacement_embeds],
        image_patch_input_indices=inputs['image_patches_indices'],
    )

    image_masked_inputs_dict = {'attention_mask': inputs['attention_mask'], 'inputs_embeds': image_masked_inputs_embeds}


    language_masked_embeds = inputs_embeds_overall.clone()

    text_slice = language_masked_embeds[:, text_ind_1+1:text_ind_2, :].clone()

    # text_noise = torch.randn_like(text_slice, device=text_slice.device, dtype=text_slice.dtype) * 3 * DEV_DICT[model_name][1]

    
    lang_replacement_embeds = torch.randn_like(text_slice) * (text_std_vector * scale_factor) + text_mean_vector


    language_masked_embeds[:, text_ind_1+1:text_ind_2, :] = lang_replacement_embeds.clone()


    language_masked_inputs_embeds = model.gather_continuous_embeddings(
        word_embeddings=language_masked_embeds,
        continuous_embeddings=[image_ori_embeddings],
        image_patch_input_indices=inputs['image_patches_indices'],
    )

    language_masked_inputs_dict = {'attention_mask': inputs['attention_mask'], 'inputs_embeds': language_masked_inputs_embeds}


    with torch.no_grad():

        cont = model.generate(**inputs, 
                                max_new_tokens=2,
                                return_dict_in_generate=True,
                                output_scores=True,
                                do_sample=False,
                                temperature=0)

        multi_logits = torch.stack(cont.scores, dim=1)
        # print('multi_logits shape:', multi_logits.shape)
        multi_probs, multi_orig_probs, output_tokens = get_choice_distributions(multi_logits, item["num_options"])


        assert int(cont.sequences[0][-1]) == int(output_tokens[0])

        
        # Take all required 


        output_text = processor.batch_decode(cont.sequences[:, -2:], skip_special_tokens=True)[0][0]

        # print('multi_probs:', multi_probs)

        # print('output_text:', output_text)



        image_single_output = model.generate(**language_masked_inputs_dict, 
                                max_new_tokens=2,
                                return_dict_in_generate=True,  # Enable returning additional outputs
                                output_scores=True,  # Request hidden states
                                do_sample=False,
                                temperature=0) 

        # Get logits and hidden states
        logits_image_single= torch.stack(image_single_output.scores, dim=1)
        # print('visual_logits shape:', visual_logits.shape)

        
        visual_probs, visual_orig_probs, visual_tokens = get_choice_distributions(logits_image_single, item["num_options"])



        text_single_output = model.generate(**image_masked_inputs_dict, 
                                max_new_tokens=2,
                                return_dict_in_generate=True,  # Enable returning additional outputs
                                output_scores=True,  # Request hidden states
                                do_sample=False,
                                temperature=0) 

        # Get logits and hidden states
        lang_logits = torch.stack(text_single_output.scores, dim=1)
        

        
        language_two_probs, language_two_orig_probs, lang_tokens = get_choice_distributions(lang_logits, item["num_options"])



        
    
    response = output_text

    # Update the GPT response in the data
    for conv in item['conversations']:
        if conv['from'] == 'gpt':
            conv['label'] = conv['value']
            conv['value'] = response
            conv['v_feature'] = vision_embeddings.to(dtype=torch.float32, device='cpu').detach().numpy().tolist()
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