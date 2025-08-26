import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb

def adjust_brightness(color, factor):
    r, g, b = to_rgb(color)
    return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0))

bins = ['0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
x = np.arange(len(bins))

phi2_data = {
    "4-bit_4-bit_corr": {
        "0.0-0.1": 113512584,
        "0.1-0.2": 30051178,
        "0.2-0.3": 9971834,
        "0.3-0.4": 2176614,
        "0.4-0.5": 393870,
        "0.5-0.6": 151661,
        "0.6-0.7": 96404,
        "0.7-0.8": 142431,
        "0.8-0.9": 674021,
        "0.9-1.0": 107562
    },
    "8-bit_8-bit_corr": {
        "0.0-0.1": 112695537,
        "0.1-0.2": 30714429,
        "0.2-0.3": 10049169,
        "0.3-0.4": 2178532,
        "0.4-0.5": 459776,
        "0.5-0.6": 156198,
        "0.6-0.7": 94977,
        "0.7-0.8": 180486,
        "0.8-0.9": 736969,
        "0.9-1.0": 11898
    },
    "16-bit_16-bit_corr": {
        "0.0-0.1": 113700468,
        "0.1-0.2": 29975863,
        "0.2-0.3": 9797087,
        "0.3-0.4": 2151660,
        "0.4-0.5": 439345,
        "0.5-0.6": 170868,
        "0.6-0.7": 72891,
        "0.7-0.8": 63039,
        "0.8-0.9": 118903,
        "0.9-1.0": 788449
    }
}

llama2_data = data = {
    "4-bit_4-bit_corr": {
        "0.0-0.1": 95907067,
        "0.1-0.2": 52790195,
        "0.2-0.3": 20399783,
        "0.3-0.4": 8134306,
        "0.4-0.5": 3205121,
        "0.5-0.6": 1020102,
        "0.6-0.7": 230859,
        "0.7-0.8": 45187,
        "0.8-0.9": 9081,
        "0.9-1.0": 14234
    },
    "8-bit_8-bit_corr": {
        "0.0-0.1": 96061233,
        "0.1-0.2": 53102541,
        "0.2-0.3": 20446794,
        "0.3-0.4": 7966487,
        "0.4-0.5": 3018671,
        "0.5-0.6": 901675,
        "0.6-0.7": 193607,
        "0.7-0.8": 40759,
        "0.8-0.9": 9862,
        "0.9-1.0": 14262
    },
    "16-bit_16-bit_corr": {
        "0.0-0.1": 94983821,
        "0.1-0.2": 52930777,
        "0.2-0.3": 20816594,
        "0.3-0.4": 8364896,
        "0.4-0.5": 3301771,
        "0.5-0.6": 1046873,
        "0.6-0.7": 239336,
        "0.7-0.8": 47529,
        "0.8-0.9": 10115,
        "0.9-1.0": 11529
    }
}

def format_value(val):
    if val >= 1_000_000:
        return f'{val/1_000_000:.1f}M'
    elif val >= 1_000:
        return f'{val/1_000:.1f}K'
    return str(val)

def y_axis_formatter(x, pos):
    return format_value(x)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12), sharey='row')

palette = sns.color_palette("colorblind", n_colors=2)
phi_color = adjust_brightness(palette[0], 0.7)
llama_color = adjust_brightness(palette[1], 0.7)

bit_titles = ["4-bit", "8-bit", "16-bit"]
phi_keys = list(phi2_data.keys())
llama_keys = list(llama2_data.keys())

for col in range(3):
    # --- Phi-2 Row ---
    ax_phi = axes[0, col]
    phi_counts = [phi2_data[phi_keys[col]][b] for b in bins]
    bars_phi = ax_phi.bar(x, phi_counts, color=phi_color, edgecolor='black')
    for bar in bars_phi:
        height = bar.get_height()
        ax_phi.text(bar.get_x() + bar.get_width() / 2, height, format_value(height), ha='center', va='bottom', fontsize=15)
    ax_phi.set_title(f"{bit_titles[col]}", fontsize=24)
    # ax_phi.set_title(f"Phi-2: {bit_titles[col]}", fontsize=20)
    ax_phi.set_xticks([])
    # ax_phi.set_xticklabels(bins, rotation=45)
    ax_phi.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    if col == 0:
        ax_phi.set_ylabel("Count", fontsize=22)

    # --- Llama-2-7B Row ---
    ax_llama = axes[1, col]
    llama_counts = [llama2_data[llama_keys[col]][b] for b in bins]
    bars_llama = ax_llama.bar(x, llama_counts, color=llama_color, edgecolor='black')
    for bar in bars_llama:
        height = bar.get_height()
        ax_llama.text(bar.get_x() + bar.get_width() / 2, height, format_value(height), ha='center', va='bottom', fontsize=15)
    # ax_llama.set_title(f"Llama-2-7B: {bit_titles[col]}", fontsize=20)
    ax_llama.set_xticks(x)
    ax_llama.set_xticklabels(bins, rotation=45)
    ax_llama.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    if col == 0:
        ax_llama.set_ylabel("Count", fontsize=22)
    ax_phi.tick_params(axis='y', labelsize=22)
    ax_llama.tick_params(axis='x', labelsize=22)
    ax_llama.tick_params(axis='y', labelsize=22)

custom_lines = [
    plt.Line2D([0], [0], color=phi_color, lw=10),
    plt.Line2D([0], [0], color=llama_color, lw=10)
]
fig.legend(custom_lines, ['Phi-2', 'Llama-2-7B'], loc='lower center', ncol=2, fontsize=18,  bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("graphs/correlation.png", dpi=300, bbox_inches='tight')
plt.show()
