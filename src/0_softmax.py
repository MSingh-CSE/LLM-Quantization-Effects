import argparse
import os

import pandas as pd
from tqdm import tqdm
import torch

from utils import load_dataset, load_tokenizer_model
from config import basepath

parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("quant", type=str, help="Quantization")
parser.add_argument("dataset_name", type=str, help="Dataset")
parser.add_argument("quant_config", type=str, help="Quantization config")
parser.add_argument("batch_size", type=str, help="Batch size")
args = parser.parse_args()

model_name = args.model_name
quant = args.quant
dataset_name = args.dataset_name
batch_size = int(args.batch_size)
quant_config = None if args.quant_config.lower() == "none" else args.quant_config.lower()

dataset = load_dataset(dataset_name)
tokenizer, model = load_tokenizer_model(model_name, quant=quant, config_q=quant_config)

def return_softmax(prompts, golds):
    "Process dataset to find softmax."
    
    results_basepath = os.path.join(basepath, "softmaxes")
    os.makedirs(results_basepath, exist_ok=True)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_logits=True)
        
    for index, p in enumerate(prompts):
        final_token_logits = outputs.logits[0][index]  # logits for the predicted token
        probs = torch.nn.functional.softmax(final_token_logits, dim=-1) 

        predicted_token_id = torch.argmax(probs).item()
        predicted_token = tokenizer.decode(predicted_token_id).strip()
        predicted_prob = probs[predicted_token_id].item()

        # Initial correctness - Changed in analysis based on different dataset
        correct = (golds[index].strip().lower() in predicted_token.lower())

        results_df = pd.DataFrame({
            'Prompt': [p],  
            'Gold': [golds[index]], 
            'Predicted_Token': [predicted_token], 
            'Predicted_Token_Prob': [predicted_prob],
            'Correct': [correct]
        })
        
        csv_path = os.path.join(results_basepath, f'{model_name}_{dataset_name}_model_softmaxes_{quant}.csv')
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, mode='w', header=True, index=False)
    
if __name__ == "__main__":

    ids = [str(i) for i in range(1, len(dataset) + 1)]
    prompts = dataset["prompt"].tolist()
    
    if dataset_name == "boolq":
        template = "Respond with True or False only.\n{}\nAnswer:"
        prompts = [template.format(p) for p in prompts]
    
    golds = dataset["gold"].tolist()

    start_index = 0
    total_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)
    pbar = tqdm(total=total_batches, desc="Processing Batches")
    
    while start_index < len(prompts):
        batch = prompts[start_index:start_index + batch_size]
        golds_batch = golds[start_index:start_index + batch_size]
        return_softmax(batch, golds_batch)
        start_index += batch_size
        pbar.update(1)

    pbar.close()
