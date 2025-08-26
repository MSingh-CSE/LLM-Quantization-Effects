import os
import argparse
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import ujson as json
import h5py
import numpy as np
from tqdm import tqdm

from config import basepath, MODEL_TO_LAYERS

parser = argparse.ArgumentParser(description="Analyze dead neurons from HDF5.")
parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("quant", type=str, choices=["32-bit", "8-bit", "4-bit"], help="Quantization type")
parser.add_argument("dataset_name", type=str, help="Dataset")
args = parser.parse_args()

model_name = args.model_name
quant = args.quant
dataset_name = args.dataset_name

h5_path = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "activation_records", "all_activations.h5")

layers = MODEL_TO_LAYERS[model_name]
print("Analyzing Layers:")
print(layers)
print()

def is_dead(activations, model_name):
    "Checks if a neuron is dead based on activation function used in model."
    
    if model_name == "opt-6b":
        return np.all(np.abs(activations) < 1e-6)
    else:
        return np.all((activations > -0.1) & (activations < 0.1))

def process_neuron(args):
    "Process each neuron's activations."
    
    layer, neuron, model_name = args
    try:
        with h5py.File(h5_path, 'r') as h5f:
            group_path = f"{layer}/{neuron}"
            if group_path not in h5f or "activations" not in h5f[group_path]:
                print(f"Error in group {layer}/{neuron}: {e}")
                return (layer, neuron, True)

            activations_ds = h5f[group_path]["activations"]
            
            is_dead_neuron = is_dead(np.concatenate(activations_ds[()]), model_name)
            return (layer, neuron, is_dead_neuron)

    except Exception as e:
        print(f"Error reading {layer}/{neuron}: {e}")
        return (layer, neuron, True)

if __name__ == "__main__":

    neuron_tasks = []
    neuron_counts = defaultdict(int)

    with h5py.File(h5_path, 'r') as h5f:
        for layer in layers:
            if layer not in h5f:
                continue
            neuron_ids = list(h5f[layer].keys())
            neuron_counts[layer] = len(neuron_ids)
            for neuron_id in neuron_ids:
                neuron_tasks.append((layer, neuron_id, model_name))


    print(f"Processing {len(neuron_tasks)} neurons with {31} workers...")
    results = []
    with Pool(processes=31) as pool:
        results = list(tqdm(pool.imap(process_neuron, neuron_tasks), total=len(neuron_tasks)))

    dead_count = defaultdict(int)
    non_dead_freq = defaultdict(list)
    dead_neurons_per_layer = defaultdict(list)
    token_counts = defaultdict(int)

    for layer, neuron, is_dead_neuron in results:
        if is_dead_neuron:
            dead_count[layer] += 1
            dead_neurons_per_layer[layer].append(neuron)

    print("\nDead Neuron Summary:")
    summary_data = {}
    for layer in layers:
        num_dead = dead_count[layer]
        total = len(h5py.File(h5_path, 'r')[layer].keys())
        print(f"Layer: {layer}")
        print(f"  Dead neurons: {num_dead}/{total}")
        print()

        output_path = os.path.join(basepath, f"{model_name}_{quant}_{dataset_name}_dead_neuron_summary.json")
        
        summary_data[layer] = {
            "dead_neurons": num_dead,
            "total_neurons": total,
        }

    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    save_path = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "dead_neurons.json")
    with open(save_path, "w") as f:
        json.dump(dead_neurons_per_layer, f, indent=4)

    print(f"Dead neurons written to: {save_path}")
