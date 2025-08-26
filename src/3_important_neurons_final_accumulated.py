import os
import json
import argparse

import csv
from tqdm import tqdm

from config import basepath, MODEL_TO_LAYERS

output_dir = os.path.join(basepath, "important_neuron_summary_csvs")
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Process model name.")

parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("quant", type=str, help="Quantization")
parser.add_argument("dataset_name", type=str, help="dataset")

args = parser.parse_args()

model = args.model_name
quantization = args.quant
dataset = args.dataset_name

dataset_results = {dataset: []}

important_neurons_basepath = os.path.join(basepath, f"{model}_{quantization}__{dataset}", "important_neurons_records")

if not os.path.exists(important_neurons_basepath):
            print(f"\n{model}-{quantization}-{dataset} attributions not available.\n")

datapoint_dirs = sorted([x for x in os.listdir(important_neurons_basepath) if os.path.isdir(os.path.join(important_neurons_basepath, x))], key=lambda k: int(k))

layer_names = MODEL_TO_LAYERS[model]

important_neurons = important_neurons = {layer: set() for layer in layer_names}

for datapoint in datapoint_dirs:
    for layer in layer_names:
        important_neuron_file_path = os.path.join(important_neurons_basepath, datapoint, layer, "important_neurons.json")

        if not os.path.exists(important_neuron_file_path):
            print(f"\n{model}-{quantization}-{dataset}-{datapoint} does not exist.\n")
            continue
        
        with open(important_neuron_file_path) as f:
            data = json.load(f)
            for key in ["best_word", "sum_based_neurons", "avg_based_neurons", "max_based_neurons"]:
                for neuron in data[key]:
                    important_neurons[layer].add(int(neuron))

row = [
    model,
    quantization,
    len(important_neurons[layer_names[0]]),
    len(important_neurons[layer_names[1]]),
    len(important_neurons[layer_names[2]])
]
dataset_results[dataset].append(row)


for dataset, rows in dataset_results.items():
    csv_path = os.path.join(output_dir, f"{dataset}_neuron_summary.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Quantization", "Initial Layer", "Middle Layer", "Last Layer"])
        writer.writerows(rows)

print(f"CSV files saved to: {output_dir}")

            