import os

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

from src.config import basepath

def calculate_ace(df, num_bins=10):
    df = df.sort_values('Predicted_Token_Prob')
    total_samples = len(df)
    samples_per_bin = total_samples // num_bins

    ace = 0.0
    bin_start = 0

    for i in range(num_bins):
        bin_end = bin_start + samples_per_bin + (1 if i < total_samples % num_bins else 0)
        bin_data = df.iloc[bin_start:bin_end]

        if not bin_data.empty:
            accuracy = bin_data['Correct'].mean()
            confidence = bin_data['Predicted_Token_Prob'].mean()
            ace += abs(accuracy - confidence)

        bin_start = bin_end

    return ace / num_bins

def process_files(input_folder):
    ace_results = []

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            try:
                base_name = os.path.splitext(file)[0]
                parts = base_name.split('_')
                if len(parts) == 5: 
                    model, dataset, _, _, quant = parts

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
                    
                    quant = '16-bit' if quant == '32-bit' else quant 
                else:
                    print(f"Skipping file with unexpected naming: {file}")
                    continue 

                df = pd.read_csv(os.path.join(input_folder, file))
                ace = calculate_ace(df)

                ace_results.append({'Model': rename_models.get(model, model), 'Dataset': rename_datasets.get(dataset, dataset), 'Quantization': quant, 'ACE': ace})
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    ace_df = pd.DataFrame(ace_results)
    ace_df.to_csv(os.path.join("final_presentation/ace_results.csv"), index=False)

    return ace_df


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
    sns.set(style='whitegrid')
    print(df.columns)

    dataset_order = ["BoolQ", "Toxicity", "Hellaswag", "PIQA", "IMDB Sentiment"]
    model_order = ["Phi-2", "Llama-2-7B", "Qwen-3B", "Qwen-7B", "Mistral-7B"]
    quant_order = ["4-bit", "8-bit", "16-bit"]

    model_palette = sns.color_palette("colorblind", n_colors=len(model_order))
    quant_shades = [0.9, 0.7, 0.5]

    # fig, axes = plt.subplots(1, len(dataset_order), figsize=(6 * len(dataset_order), 6), sharey=True)
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(6 * len(dataset_order), 6), sharey=True)
    fig.subplots_adjust(wspace=0.05, bottom=0.2)  # Make space for two rows of legend


    if len(dataset_order) == 1:
        axes = [axes]

    for i, dataset in enumerate(dataset_order):
        ax = axes[i]
        subset_df = df[df['Dataset'] == dataset].copy()
        bars_data = []

        for m_idx, model in enumerate(model_order):
            for q_idx, quant in enumerate(quant_order):
                row = subset_df[(subset_df['Model'] == model) & (subset_df['Quantization'] == quant)]
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
            # Add text labels
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

    model_patches = [
        mpatches.Patch(color=model_palette[i], label=model_order[i])
        for i in range(len(model_order))
    ]
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

    input_folder = os.path.join(basepath, "softmaxes")
    ace_df = process_files(input_folder)

    plot_grouped_bars(ace_df, value_col='ACE', ylabel='Adaptive Calibration Error (ACE)', save_path='graphs/ace_barplot.png')

