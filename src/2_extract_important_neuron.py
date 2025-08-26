import os
import json
import numpy as np
import argparse

import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from config import basepath

parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("quant", type=str, help="Quantization settings")
parser.add_argument("dataset_name", type=str, help="Name of the dataset")

args = parser.parse_args()

model_name = args.model_name
quant = args.quant
dataset_name = args.dataset_name
attributions_basepath = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "attributions_records")
important_neurons_basepath = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "important_neurons_records")

def normalize_attribution(attr_np_array):
    """
    Normalizes the attribution array.
    """
    attr_np_array = np.abs(attr_np_array) 
    max_score = np.max(attr_np_array)
    if max_score > 0:
        attr_np_array /= max_score     
    return attr_np_array

def imp_neurons_after_threshold(attr_np_array, threshold=0.7):
    "Find top contributing neurons."
    attr_np_array = normalize_attribution(attr_np_array)
    if np.sum(attr_np_array) <= 0:
        return []
    
    indices_above_threshold = np.where(attr_np_array > threshold)[0]
    sorted_indices_of_filtered = indices_above_threshold[np.argsort(attr_np_array[indices_above_threshold])[::-1]]
    return sorted_indices_of_filtered.tolist()

def process_datapoint(args):
    ""
    
    attributions_basepath, datapoint, layer, important_neurons_basepath = args
    attributions_file_path = os.path.join(attributions_basepath, datapoint, layer, 'attributions.json')

    output_file = os.path.join(important_neurons_basepath, datapoint, layer, "important_neurons.json")

    if os.path.exists(output_file):
        print(f"{output_file} already exists.")

    with open(attributions_file_path) as f:
        data = json.load(f)

    seq_attribution = data["seq_attribution"]
    input_tokens = list(seq_attribution.keys())
    
    important_input_token = max(seq_attribution, key= lambda x: seq_attribution[x])
    
    neurons_attributions = data["neurons_attributions"]
    output_tokens = neurons_attributions.keys()
    output_word = "".join(output_tokens)
    
    important_neurons = {
        "datapoint": datapoint,
        "layer_name": layer,
        "input_tokens": input_tokens,
        "input_token_with_max_attr": important_input_token,
        "output_word": output_word
    }

    attribution_output_word = {
        output_word: {}
    }

    for output_token in neurons_attributions:
        for input_token in neurons_attributions[output_token]:
            attribution_numpy_array = np.array(neurons_attributions[output_token][input_token])
            attribution_output_word[output_word].setdefault(input_token, np.zeros_like(attribution_numpy_array))
            attribution_output_word[output_word][input_token] += attribution_numpy_array
    
    sum_test = 0
    for output_token in neurons_attributions:
        for input_token in neurons_attributions[output_token]:
            sum_test += neurons_attributions[output_token][input_token][0]
            break
    
    # Most attributed token based
    # Given below attributions for mulitple input tokens
    # np.array([0.1, 0.2, 0.3])
    # np.array([0.4, 0.1, 0.2])
    # np.array([0.3, 0.4, 0.1])
    # np.array([0.2, 0.5, 0.4]) # Suppose best word attr
    # Attribution for important word: [0.2 0.5 0.4]
    # Important neurons for best word: [1]
    attribution_for_best_word = attribution_output_word[output_word][important_input_token]
    important_neurons["best_word"] = imp_neurons_after_threshold(attribution_for_best_word)

    layer_shape = attribution_output_word[output_word][important_input_token].shape

    # Input sequence based
    # np.abs addition of all the attributions across input tokens for output word
    # to find the final important neuron for complete word
    # Given below attributions for mulitple input tokens
    # np.array([0.1, 0.2, 0.3])
    # np.array([0.5, 0.0, 0.2])
    # np.array([0.5, 0.4, 0.1])
    # np.array([0.2, 0.5, 0.4])
    # Attribution based on abs sum: [1.3 1.1 1. ]
    # Important neurons based on sum across all tokens: [0, 1]
    sum_attributions_across_input_tokens = np.zeros(layer_shape)
    for input_token in attribution_output_word[output_word]:
        sum_attributions_across_input_tokens += np.abs(attribution_output_word[output_word][input_token])

    important_neurons["sum_based_neurons"] = imp_neurons_after_threshold(sum_attributions_across_input_tokens)

    # Average of attributions using input tokens
    # Given below attributions for mulitple input tokens
    # np.array([0.1, 0.2, 0.3])
    # np.array([0.5, 0.0, 0.2])
    # np.array([0.5, 0.4, 0.1])
    # np.array([0.2, 0.5, 0.4])
    # Attribution based on avg of sum: [0.325 0.275 0.25 ]
    # Important neurons based on average across all tokens: [0, 1]
    avg_attribution = sum_attributions_across_input_tokens/len(input_tokens)
    important_neurons["avg_based_neurons"] = imp_neurons_after_threshold(avg_attribution)

    # For each input token which neuron is important select that neuron
    # List of all attributions: [array([0.1, 0.2, 0.3]), array([0.4, 0.1, 0.2]), array([0.3, 0.4, 0.1]), array([0.2, 0.5, 0.4])]
    # Stacked Attributions (2D Array):
    #  [[0.1 0.2 0.3]
    #  [0.4 0.1 0.2]
    #  [0.3 0.4 0.1]
    #  [0.2 0.5 0.4]]
    # Maximum values of attributions across input tokens: [0.4 0.5 0.4]
    attributions_for_all_input_tokens = list(attribution_output_word[output_word].values()) # list of all attributions
    stacked_attributions = np.stack(attributions_for_all_input_tokens) # 2D array with attributions
    max_values_attributions = np.amax(stacked_attributions, axis=0) # selecting max across rows

    important_neurons["max_based_neurons"] = imp_neurons_after_threshold(max_values_attributions)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(important_neurons, f, indent=4)
        
def collect_tasks(datapoint_dirs):
    tasks = []
    for datapoint in datapoint_dirs:
        layers = os.listdir(os.path.join(attributions_basepath, datapoint))
        tasks.extend([(attributions_basepath, datapoint, layer, important_neurons_basepath) for layer in layers])
    return tasks

if __name__ == "__main__":

    datapoint_dirs = sorted([x for x in os.listdir(attributions_basepath) if os.path.isdir(os.path.join(attributions_basepath, x))], key=lambda k: int(k))

    tasks = collect_tasks(datapoint_dirs)
    
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_datapoint, tasks), total=len(tasks), colour="yellow", desc="Processing Layers", leave=False))
