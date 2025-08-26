# Change paths to actual downloaded model locations or if open internet allowed - to huggingface model URL.
MODELS = {
        "phi-2": {
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/phi-2",
        },
        "llama-7b": {
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/Llama-2-7b-chat-hf",
        },
        "mistral-7b": {
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/Mistral-7B-Instruct-v0.3",
        }, 
        "qwen-3b": {
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/Qwen2.5-3B-Instruct",
        }, 
        "qwen-7b": {
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/Qwen2.5-7B-Instruct",
        },
        "opt-6b":{
            "path": "/home/mn308829/projects/def-hsajjad/models/huggingface_cache/opt-6.7b",
        },
    }

DATASETS = {
    "boolq":{
        "path": "datasets/boolq.csv"
    },
    "piqa": {
        "path": "datasets/piqa.csv"
    },
    "toxic": {
        "path": "datasets/toxicity.csv"
    },
    "hellaswag": {
        "path": "datasets/hellaswag.csv"
    },
    "sentiment": {
        "path": "datasets/IMDB_sentiment.csv"
    },
}

MODEL_TO_LAYERS = {
    "phi-2": [f"model.layers.{i}.mlp.activation_fn" for i in [0, 14, 31]],
    "llama-7b": [f"model.layers.{i}.mlp.act_fn" for i in [0, 14, 31]],
    "opt-6b": [f"model.decoder.layers.{i}.activation_fn" for i in [0, 14, 31]],
    "mistral-7b": [f"model.layers.{i}.mlp.act_fn" for i in [0, 14, 31]],
    "qwen-3b": [f"model.layers.{i}.mlp.act_fn" for i in [0, 14, 35]],
    "qwen-7b": [f"model.layers.{i}.mlp.act_fn" for i in [0, 14, 27]],
}

basepath = "results"
