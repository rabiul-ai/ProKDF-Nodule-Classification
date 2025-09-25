import numpy as np
import matplotlib.pyplot as plt

output_path = 'Code Outputs/p28 comparison graph among 4 views'

# Define the data for models
models = ["1V", "3V", "6V", "9V"]
x = np.arange(len(models))  # X-axis positions

mean_scores = {
    "Accuracy": [80.28, 85.36, 85.04, 87.06],
    "F1-score": [80.56, 85.76, 85.47, 87.48],
    "AUC": [80.40, 89.90, 90.65, 92.00],
}

# Define standard deviations (for error bars)
std_scores = {
    "Accuracy": [3.49, 2.25, 2.48, 2.29],
    "F1-score": [3.15, 1.93, 2.04, 2.09],
    "AUC": [3.53, 2.04, 1.79, 1.36],
}

# Use Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"

# Create subplots for 3 metrics in a single row
fig, axes = plt.subplots(1, 3, figsize=(7.1, 7.1/3))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors

# Plot each metric
for i, (ax, (metric, values), color) in enumerate(zip(axes, mean_scores.items(), colors)):
    ax.bar(
        x, values,
        yerr=std_scores[metric],
        color=color,
        alpha=0.85,
        error_kw=dict(capsize=0),
        label=metric
    )
    
    # Customize axis
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(70, 95)

    if i == 0:
        ax.set_ylabel("Score (mean ± std.) (%)")
    
    # Remove top and right spines (rectangle border)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set grid style: thin dashed lines with low opacity
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Ensure grid lines are below the bars

# Add legend outside top of the plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

# Adjust the layout to prevent clipping of the legend
plt.tight_layout(pad=1.0)  # Increase padding between subplots

# Save the figure with enough space for the legend
plt.savefig(f'{output_path}/view_comparison2.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

plt.show()
