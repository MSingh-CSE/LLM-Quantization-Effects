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
batch="32"

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

> Manpreet Singh and Hassan Sajjad. "Interpreting the Effects of Quantization on LLMs." IJCNLP-AACL 2025, Mumbai, India. pp. 2267–2281.

Bibtex:
```bibtex
@inproceedings{singh-sajjad-2025-interpreting,
    title = "Interpreting the Effects of Quantization on {LLM}s",
    author = "Singh, Manpreet  and
      Sajjad, Hassan",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.123/",
    pages = "2267--2281",
    ISBN = "979-8-89176-298-5",
    abstract = "Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique."
}
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
