import os
import json
import argparse

from tqdm import tqdm
import numpy as np
import h5py
import torch

from utils import load_dataset
from config import basepath, MODEL_TO_LAYERS
from utils import load_tokenizer_model

def check_h5_corruption(h5_path):
    "Check h5 file corruption."
    try:
        with h5py.File(h5_path, 'r') as f:
            _ = list(f.keys())
        return False
    except Exception as e:
        print(f"[Warning] Corruption detected in {h5_path}: {e}")
        return True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("quant", type=str, help="Quantization")
    parser.add_argument("dataset_name", type=str, help="Dataset")
    parser.add_argument("quant_config", type=str, help="Quantization config")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    model_name = args.model_name
    quant = args.quant
    dataset_name = args.dataset_name
    quant_config = None if args.quant_config.lower() == "none" else args.quant_config.lower()
    batch_size = args.batch_size

    layer_names = MODEL_TO_LAYERS[model_name]
    activations_basepath = os.path.join(basepath, f"{model_name}_{quant}__{dataset_name}", "activation_records")
    h5_path = os.path.join(activations_basepath, "all_activations.h5")
    status_path = os.path.join(activations_basepath, "completed_batches.json")
    string_dt = h5py.string_dtype(encoding='utf-8')


    os.makedirs(activations_basepath, exist_ok=True)

    if os.path.exists(h5_path) and check_h5_corruption(h5_path):
        print("[Warning] Corrupted all_activations.h5. Deleting.")
        os.remove(h5_path)

    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            completed_batches = json.load(f)
    else:
        completed_batches = {}

    tokenizer, model = load_tokenizer_model(model_name, quant, extract_activation=True)
    model.eval()

    dataset = load_dataset(dataset_name)
    sentences = dataset['prompt'].tolist()

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_activation(name)))

    total_batches = (len(sentences) + batch_size - 1) // batch_size

    with h5py.File(h5_path, 'w') as h5file:
        for batch_number, i in tqdm(enumerate(range(0, len(sentences), batch_size)), total=total_batches):

            if str(batch_number) in completed_batches:
                print(f"[✓] Skipping batch {batch_number} (already completed)")
                continue

            batch_sentences = sentences[i:i + batch_size]
            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to('cuda')

            with torch.no_grad():
                model(**inputs)

            input_ids = inputs["input_ids"].cpu()
            attention_mask = inputs["attention_mask"].cpu()
            dt = h5py.string_dtype(encoding='utf-8')
            token_lists = [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]

            for layer_name in layer_names:
                layer_output = activations[layer_name].numpy()
                if layer_name in h5file:
                    layer_group = h5file[layer_name]
                else:
                    layer_group = h5file.create_group(layer_name)

                for neuron_id in range(layer_output.shape[2]):
                    n_id = str(neuron_id)
                    if n_id in layer_group:
                        neuron_group = layer_group[n_id]
                    else:
                        neuron_group = layer_group.create_group(n_id)

                    activation_dtype = layer_output[0,0,0].dtype

                     # === Tokens dataset ===
                    if 'tokens' not in neuron_group:
                         neuron_group.create_dataset(
                                                        'tokens',
                                                        shape=(0,),
                                                        maxshape=(None,),
                                                        dtype=h5py.vlen_dtype(string_dt),
                                                        chunks=True
                                                    )

                    # === Activations dataset ===
                    if 'activations' not in neuron_group:
                        neuron_group.create_dataset(
                                                        'activations',
                                                        shape=(0,),
                                                        maxshape=(None,),
                                                        dtype=h5py.vlen_dtype(activation_dtype),
                                                        chunks=True
                                                    )
                        neuron_group.attrs['activation_dtype'] = str(activation_dtype)

                    token_ds = neuron_group['tokens']
                    act_ds = neuron_group['activations']

                    valid_tokens = []
                    valid_activations = []

                    for b_idx, (tokens, mask) in enumerate(zip(token_lists, attention_mask)):
                        sent_tokens = []
                        sent_acts = []
                        for t_idx, keep in enumerate(mask):
                            if keep:
                                sent_tokens.append(str(tokens[t_idx]))
                                sent_acts.append(layer_output[b_idx, t_idx, neuron_id])
                        valid_tokens.append(sent_tokens)
                        valid_activations.append(np.array(sent_acts, dtype=activation_dtype))

                    n_new = len(valid_tokens)
                    tok_old = token_ds.shape[0]
                    valid_tokens_np = [np.array(vt, dtype=string_dt) for vt in valid_tokens] 
                    for j, token_array in enumerate(valid_tokens_np):
                        token_ds.resize((tok_old + j + 1,))
                        token_ds[tok_old + j] = token_array

                    act_old = act_ds.shape[0]
                    valid_activations_np = np.array(valid_activations, dtype=object)
                    for j, activation_array in enumerate(valid_activations_np):
                        act_ds.resize((act_old + j + 1,))
                        act_ds[act_old + j] = activation_array


    for hook in hooks:
        hook.remove()

    print("All processing complete.")
