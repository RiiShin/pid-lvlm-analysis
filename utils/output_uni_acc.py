import json
import argparse
import numpy as np
import os

def calculate_accuracy(file_path):
    """
    Reads a JSON file, computes predictions from probabilities,
    and calculates the accuracy against the provided labels.

    Args:
        file_path (str): The path to the input JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    correct_predictions = 0
    total_predictions = 0
    all_options = ['A', 'B', 'C', 'D'] # Extended list just in case

    for item in data:
        # Find the conversation turn from 'gpt'
        gpt_turn = next((turn for turn in item.get("conversations", []) if turn.get("from") == "gpt"), None)
        
        # Get the number of valid options for this specific question
        num_options = item.get("num_options", 4) # Default to 4 if not present
        # print(f"Number of options for this item: {num_options}")
        current_options = all_options[:num_options]

        if gpt_turn and "label" in gpt_turn and "l_prob" in gpt_turn:
            label = gpt_turn["label"]
            l_prob = gpt_turn["l_prob"]

            if not isinstance(l_prob, list) or not l_prob:
                continue
            
            # Ensure we only look at probabilities corresponding to valid options
            # (Assuming l_prob length matches or exceeds num_options, we slice it)
            valid_probs = l_prob[:num_options]

            # Check for uniform distribution (all probabilities are equal)
            # Using np.isclose to handle floating point comparisons safely
            if np.all(np.isclose(valid_probs, valid_probs[0])):
                # Uniform distribution detected, count as false prediction
                total_predictions += 1
                continue

            # Get the index of the highest probability within valid options
            pred_index = np.argmax(valid_probs)

            # Map the index to the corresponding option letter
            if pred_index < len(current_options):
                prediction = current_options[pred_index]
            else:
                continue

            # Compare prediction with the ground truth label
            if prediction == label:
                correct_predictions += 1
            
            total_predictions += 1

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) 
        print(f"Total predictions: {total_predictions}")
        # print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No valid predictions could be made from the provided data.")

if __name__ == "__main__":
    
    directory = '/work/gw42/w42010/msra/all_new_type/results/embeds/pmc/val'
    print(f"Processing directory: {directory}")
    
    # Iterate over all files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            print(f"Calculating accuracy for file: {filename}")
            calculate_accuracy(path)
            print("-" * 30)


    # directory = '/home/xiu.lixin/msra/all_new_codes/results/embeddings/mmbench/val'
    # print(f"Processing directory: {directory}")
    # filename = sorted(os.listdir(directory))[0]
    # path = os.path.join(directory, filename)
    # print(f"Calculating accuracy for file: {filename}")
    # calculate_accuracy(path)
    