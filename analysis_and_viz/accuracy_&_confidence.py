import os
import glob

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from src.config import basepath

def process_csv(file, name):
    "Process input csvs."

    df = pd.read_csv(file, dtype={'Prompt': 'string',
                                  'Gold': 'string', 
                                  'Predicted_Token': 'string', 
                                  'Predicted_Token_Prob': 'float', 
                                  'Correct': 'int'})

    parts = name.split('_')
    model = parts[0]
    dataset = parts[1]
    quantization = parts[-1].replace(".csv", "")

    rename_models = {
        "phi-2": "Phi-2",
        "llama-7b": "Llama-2-7B",
        "qwen-3b": "Qwen-3B",
        "qwen-7b": "Qwen-7B",
        "mistral-7b": "Mistral-7B"
    }

    rename_datasets = {
        "toxic": "Toxicity",
        "boolq": "BoolQ",
        "hellaswag": "Hellaswag",
        "piqa": "PIQA",
        "sentiment": "IMDB Sentiment"
    }

    df['model'] = rename_models.get(model, model)
    df['dataset'] = rename_datasets.get(dataset, dataset)
    df['quantization'] = quantization
    df['Prompt'] = df['Prompt'].str.strip()
    
    return df

def run_analysis(data):
    "Calculate accuracy and average confidence."
    
    results = []
    for model in data['model'].unique():
        for dataset in data['dataset'].unique():
            for quantization in data['quantization'].unique():
                subset = data[(data['model'] == model) & (data['dataset'] == dataset) & (data['quantization'] == quantization)]
                
                if not subset.empty:  # Only calculate if subset is not empty
                    accuracy = np.mean(subset['Correct'].values)
                    average_confidence = np.mean(subset['Predicted_Token_Prob'].values)
                    
                    results.append({
                        'model': model,
                        'dataset': dataset,
                        'quantization': quantization,
                        'accuracy': accuracy,
                        'average_confidence': average_confidence
                    })
    
    return pd.DataFrame(results)

def adjust_brightness(color, factor):
        """Return a lightened or darkened color"""
        r, g, b = to_rgb(color)
        return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0))

def plot_grouped_bars(
        df, 
        value_col='accuracy',
        ylabel='Accuracy',
        save_path='grouped_accuracy.png',
    ):

        dataset_order = ["BoolQ", "Toxicity", "Hellaswag", "PIQA", "IMDB Sentiment"]
        model_order = ["Phi-2", "Llama-2-7B", "Qwen-3B", "Qwen-7B", "Mistral-7B"]
        quant_order = ["4-bit", "8-bit", "16-bit"]

        model_palette = sns.color_palette("colorblind", n_colors=len(model_order))
        quant_shades = [0.9, 0.7, 0.5]

        fig, axes = plt.subplots(1, len(dataset_order), figsize=(6 * len(dataset_order), 6), sharey=True)
        fig.subplots_adjust(wspace=0.05, bottom=0.2)  # Make space for two rows of legend


        if len(dataset_order) == 1:
            axes = [axes]

        for i, dataset in enumerate(dataset_order):
            ax = axes[i]
            subset_df = df[df['dataset'] == dataset].copy()
            bars_data = []

            for m_idx, model in enumerate(model_order):
                for q_idx, quant in enumerate(quant_order):
                    row = subset_df[(subset_df['model'] == model) & (subset_df['quantization'] == quant)]
                    value = row[value_col].values[0] if not row.empty else 0.0
                    bars_data.append({
                        'Model': model,
                        'Quantization': quant,
                        'Value': value,
                        'ModelIndex': m_idx,
                        'QuantIndex': q_idx
                    })

            bars_df = pd.DataFrame(bars_data)

            total_models = len(model_order)
            total_quants = len(quant_order)
            bar_width = 0.5 
            group_width = bar_width * total_quants
            spacing = 0.12  

            positions = []
            for idx, row in bars_df.iterrows():
                base = row['ModelIndex'] * (group_width + spacing)
                offset = (row['QuantIndex'] - 1) * bar_width
                positions.append(base + offset)
            bars_df['Position'] = positions

            for q_idx, quant in enumerate(quant_order):
                quant_rows = bars_df[bars_df['Quantization'] == quant]
                colors = [adjust_brightness(model_palette[i], quant_shades[q_idx]) for i in quant_rows['ModelIndex']]
                bars = ax.bar(
                    quant_rows['Position'],
                    quant_rows['Value'],
                    width=bar_width,
                    color=colors,
                    label=quant,
                    edgecolor = "none"
                )
                # for bar in bars:
                #     height = bar.get_height()
                #     ax.text(
                #         bar.get_x() + bar.get_width()/2,
                #         height + 0.005,
                #         f'{height:.2f}',
                #         ha='center',
                #         va='bottom',
                #         fontsize=7.5,
                #     )

            ax.set_title(dataset, fontsize=24)
            ax.set_ylabel(ylabel if i == 0 else "", fontsize=22)
            ax.set_xticks([
                m * (group_width + spacing) for m in range(total_models)
            ])
            ax.set_xticklabels(model_order, rotation=15, fontsize=15)
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(True, linestyle='--', alpha=0.5)


        quant_patches = [
            mpatches.Patch(color=adjust_brightness("gray", quant_shades[i]), label=quant_order[i])
            for i in range(len(quant_order))
        ]

        fig.legend(
            handles=quant_patches,
            title="Quantization",
            loc='lower center',
            bbox_to_anchor=(0.5, -0.10),
            fontsize=18,
            ncols=len(quant_order),
            title_fontproperties={'size': 18, 'weight': 'bold'},
            frameon=False
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    softmax_path = os.path.join(basepath, "softmaxes")

    csv_files = [(os.path.join(softmax_path, f), f) for f in os.listdir(softmax_path) if f.endswith(".csv")]
    data = pd.concat([process_csv(path, file_name) for (path, file_name) in csv_files])

    # Ordering 4 -> 8-bit -> 16-bit in graphs
    data['quantization'] = pd.Categorical(data['quantization'], categories=["4-bit", "8-bit", "16-bit"], ordered=True)

    dict_results = run_analysis(data)
    dict_results["quantization"] = pd.Categorical(dict_results['quantization'], ["4-bit", "8-bit", "16-bit"])
    dict_results = dict_results.sort_values(by='quantization')

    model_order = ["Phi-2", "Llama-2-7B", "Qwen-3B", "Qwen-7B", "Mistral-7B"]
    datasets = ["BoolQ", "Toxicity", "Hellaswag", "PIQA", "IMDB Sentiment"]
    dataset_colors = {
        "BoolQ": "#1f77b4",         # Blue
        "Toxicity": "#ff7f0e",      # Orange
        "Hellaswag": "#2ca02c",     # Green
        "PIQA": "#d62728",          # Red
        "IMDB Sentiment": "#9467bd" # Purple
    } 

    plot_grouped_bars(dict_results, value_col='accuracy', ylabel='Accuracy', save_path='graphs/accuracy_barplot.png')
    plot_grouped_bars(dict_results, value_col='average_confidence', ylabel='Average Confidence', save_path='graphs/confidence_barplot.png')
