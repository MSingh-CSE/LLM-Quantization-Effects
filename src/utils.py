import re
import random

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

from config import MODELS, DATASETS

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True
    )

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

def get_model_path(model_name, quant, config_q):
    """Return the model path based on the quantization level and configuration."""
    
    model_path = MODELS[model_name]["path"]
    
    print(f"[DEBUG] Loading model '{model_name}' with quantization '{quant}' and config_q '{config_q}'")
    print(f"[DEBUG] Model path: {model_path}")
    
    return model_path


def load_model_with_quant(model_name, path, quant, extract_activation, config_q):
    """Load the model based on quantization level and configuration."""

    kwargs = {"device_map": "auto"}
    
    if extract_activation:
        kwargs["output_hidden_states"] = True

    print(f"[DEBUG] Loading model from path: {path}")
    print(f"[DEBUG] Quantization: {quant}, Extract Activation: {extract_activation}, Config_q: {config_q}")
    
    if quant == "16-bit":
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    elif quant == "8-bit" and config_q is None:
        print("[DEBUG] Using 8-bit quantization (bnb_config)")
        return AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, **kwargs)
    elif quant == "4-bit" and config_q is None:
        print("[DEBUG] Using 4-bit quantization (nf4_config)")
        return AutoModelForCausalLM.from_pretrained(path, quantization_config=nf4_config, **kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)

def load_tokenizer_model(model_name, quant=None, extract_activation=False, config_q=None):
    """Load the tokenizer and model based on the given parameters."""

    path = get_model_path(model_name, quant, config_q)
    model = load_model_with_quant(model_name, path, quant, extract_activation, config_q)

    if model_name == "opt-6b":
        print("[DEBUG] Loading tokenizer for 'opt-6b' with use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    else:
        print(f"[DEBUG] Loading tokenizer for '{model_name}'")
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    print(model.config)
    print(f"[DEBUG] Model and tokenizer for '{model_name}' loaded successfully")
    
    return tokenizer, model


def load_dataset(dataset_name):
    """Load dataset."""

    path = DATASETS[dataset_name]["path"]

    def load_csv(path):
        return pd.read_csv(path, dtype=str)

    dataset = load_csv(path)
    
    return dataset