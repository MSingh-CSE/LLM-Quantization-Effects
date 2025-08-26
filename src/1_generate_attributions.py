import os
import json
import argparse

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from captum.attr import (
    LayerIntegratedGradients, 
    LLMGradientAttribution,
    TextTokenInput,
    )

from utils import load_tokenizer_model, load_dataset
from config import basepath, MODEL_TO_LAYERS

parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("quant", type=str, help="Quantization")
parser.add_argument("dataset_name", type=str, help="dataset")
parser.add_argument("batch_size", type=str, help="Batch size")
parser.add_argument("quant_config", type=str, help="Quantization config")

args = parser.parse_args()

model_name = args.model_name
quant = args.quant
dataset_name = args.dataset_name
batch_size = int(args.batch_size)
quant_config = None if args.quant_config.lower() == "none" else args.quant_config.lower()

attributions_basepath = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "attributions_records")

def extract_relevant_layers(model, model_name):
    """
    Extracts relevant layers from the given model based on the model name.

    Args:
    - model (torch.nn.Module): Modle in nn.Module format.
    - model_name (str): Name of the model.
    
    Returns:
    - dict: A dictionary of layer names and layer modules.
    """
    layer_dict = {}

    for name, module in model.named_modules():
        if name in MODEL_TO_LAYERS[model_name]: 
            layer_dict[name] = module
    return layer_dict

def generate_output(model, tokenizer, prompts, basepath):
    """
        Prompts the given model.
    """
    inputs = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=1)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_outputs = []
    
    for input, output in zip(prompts, generated_texts):
        cleaned_output = output[len(input):]
        cleaned_outputs.append(cleaned_output.strip())

    results_df = pd.DataFrame({
        'Prompt': prompts,
        'Generated Output': cleaned_outputs
    })

    csv_path = os.path.join(basepath,'model_generation.csv')
    os.makedirs(basepath, exist_ok=True)
    if os.path.exists(csv_path):
        results_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_path, mode='w', header=True, index=False)

    return cleaned_outputs

def process_and_save_attributions(model, tokenizer, dataset, basepath):
    """
    Iterate through dataset to calculate the attributions for given layers.
    """
    ids = [str(i) for i in range(1, len(dataset)+1)]
    prompts = dataset["prompt"].tolist()
    if dataset_name == "boolq":
        template = "Respond with True or False only.\n{}\nAnswer:"
        prompts = [template.format(p) for p in prompts]

    outputs = []
    start_index = 0
    total_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)
    pbar = tqdm(total=total_batches, desc="Processing Inference")

    while start_index < len(prompts):
        batch = prompts[start_index:start_index+batch_size]
        outputs.extend(generate_output(model, tokenizer, batch, basepath))
        start_index += batch_size
        pbar.update(1)
    
    pbar.close()

    valid_outputs = [str(iij).lower() for iij in dataset["gold"].unique()]

    for id, prompt, output in tqdm(zip(ids, prompts, outputs), total=len(prompts), colour="blue", desc="datapoint: ", leave=False):
        datapoint_path = os.path.join(basepath, id)
        if output.lower() not in valid_outputs:
            print(f"\n[ERROR] {id} output is -|{output}|-")
        if os.path.exists(datapoint_path):
            print(f"\nDatapoint {id} record already present.")
        else:
            process_layer(model, tokenizer, prompt, output, datapoint_path)

def process_layer(model, tokenizer, prompt, output, datapoint_path):
    """
    Calculating attributions for given layers.
    """
    for layer_name, layer in tqdm(relevant_layers.items(), colour="red", desc="Layer: ", leave=False):
        ig = LayerIntegratedGradients(model, layer)
        llm_attr = LLMGradientAttribution(ig, tokenizer)
        inp = TextTokenInput(prompt, tokenizer)
        neuron_attr_wrt_output, attr_res = llm_attr.attribute(inp, target=str(output))

        layer_path = os.path.join(datapoint_path, layer_name)
        save_attributions(attr_res, neuron_attr_wrt_output, layer_path)

def save_attributions(attr_res, neuron_attr_wrt_output, layer_path):
    """
    Save attributions as JSON file.
    """
    output_path = os.path.join(layer_path, "attributions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # convert torch tensors to lists for JSON
    for output_token in neuron_attr_wrt_output:
        for input_token in neuron_attr_wrt_output[output_token]:
            neuron_attr_wrt_output[output_token][input_token] = neuron_attr_wrt_output[output_token][input_token].tolist()

    final_neuron_attribution = {
        "neurons_attributions": neuron_attr_wrt_output,
        "token_attributions": attr_res.token_attr.tolist(),
        "seq_attribution": attr_res.seq_attr_dict
    }

    with open(output_path, 'w') as f:
        json.dump(final_neuron_attribution, f, indent=4)
    torch.cuda.empty_cache()

if __name__ =="__main__":

    dataset = load_dataset(dataset_name)
    tokenizer, model = load_tokenizer_model(model_name, quant)
    relevant_layers = extract_relevant_layers(model, model_name)
    
    layers_to_store = {key:relevant_layers[key].__str__() for key in relevant_layers}
    
    file_path = os.path.join(attributions_basepath,  f"{model_name}_extracted_layers.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
            json.dump(layers_to_store, json_file, indent=4)

    process_and_save_attributions(model, tokenizer, dataset, attributions_basepath)
            
            


    

