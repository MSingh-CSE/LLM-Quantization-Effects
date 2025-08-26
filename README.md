# Interpreting Effects of Quantization on LLMs
 This repo contains code for paper: "Interpreting the Effects of Quantization on LLMs"

 ## Environment

Step 1: Create and activate virtual environment
```bash
python -m venv env_name && source env_name/bin/activate
```

Step 2: Install the dependencies
```bash
pip install -r requirements.txt
```

## Modules
> Note: 
> Before executing modules please update the models path in `src\config.py` to correct location of models on your system or provide huggingface model URL path (if open internet allowed).

### Confidence & Calibration
> Edit model, quantization or dataset as required.

```bash
models=("phi-2" "llama-7b" "qwen-3b" "qwen-7b" "mistral-7b")
quantizations=("4-bit" "8-bit" "32-bit")
datasets=("boolq" "toxic" "piqa" "hellaswag" "sentiment")
quant_config="none"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for quant in "${quantizations[@]}"; do
      python src/0_softmax.py $model $quant $dataset $quant_config $batch
    done
  done
done
```

Execute below once softmaxes are processed for all the configs:
```bash
python -m analysis_and_viz.accuracy_\&_confidence
python -m ace
```
> Note: For certain datasets or models, we apply specific adjustments to matching condition before calculating accuracy.



### Salient Neurons
> Note: Replace `llm_attr.py` provided with captum with `llm_attr.py` provided in the repo.

> Edit model, quantization or dataset as required.

```bash
models=("phi-2" "llama-7b" "qwen-3b" "qwen-7b" "mistral-7b")
quantizations=("4-bit" "8-bit" "32-bit")
datasets=("boolq" "toxic" "piqa" "hellaswag" "sentiment")
quant_config="none"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for quant in "${quantizations[@]}"; do
      python src/1_generate_attributions.py "$model" "$quant" "$dataset" 32 "$quant_config"
      python src/2_extract_important_neuron.py "$model" "$quant" "$dataset"
      python src/3_important_neurons_final_accumulated.py "$model" "$quant" "$dataset"
    done
  done
done
```

### Dead Neurons
> Edit model, quantization or dataset as required.

```bash
models=("phi-2" "llama-7b" "qwen-3b" "qwen-7b" "mistral-7b")
quantizations=("4-bit" "8-bit" "32-bit")
datasets=("boolq" "toxic" "piqa" "hellaswag" "sentiment")
quant_config="none"
batch="32"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for quant in "${quantizations[@]}"; do
      python src/4_generate_activations_h5py.py "$model" "$quant" "$dataset" "$quant_config" --batch $batch
      python src/5_dead_neurons.py "$model" "$quant" "$dataset"
    done
  done
done
```

### Correlation Plot
> Neurons combination count already processed with activations for Phi-2 and Llama-2-7b and are available within python file.
```bash
python analysis_and_viz/correlation.py
```

## Cite This Work

If you use this code in your research, please cite our paper:

> Manpreet Singh, & Hassan Sajjad. (2025). Interpreting the Effects of Quantization on LLMs. [https://arxiv.org/abs/2508.16785]

Bibtex:
```bibtex
@misc{singh2025interpretingeffectsquantizationllms,
      title={Interpreting the Effects of Quantization on LLMs}, 
      author={Manpreet Singh and Hassan Sajjad},
      year={2025},
      eprint={2508.16785},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.16785}, 
}
```

## Acknowledgements

This project uses the following models and datasets:

*Models*
- **Phi-2** (Javaheripi and Bubeck, 2023)
- **Llama-2 7B** (Touvron et al., 2023)
- **Qwen 2.5 3B and 7B** (Qwen et al., 2025)
- **Mistral-7B** (Jiang et al., 2023)

*Datasets*
- **BoolQ** (Clark et al., 2019)
- **Jigsaw Toxicity dataset** (cjadams et al., 2017)
- P**hysical Interaction: Question Answering (PIQA)** (Bisk et al., 2020)
- **Hellaswag** (Zellers et al., 2019)
- **IMDB sentiment**
classification (Maas et al., 2011)




Please make sure to cite the original papers if you use this repository in your research.