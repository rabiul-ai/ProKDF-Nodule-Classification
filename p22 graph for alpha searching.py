import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


output_path = 'Code Outputs/p22 graph for alpha searching'


# Define the data (Parsing manually as given in text format)
data_dict = {
    "Alpha": np.linspace(0, 1, 21),
    "Mean F1 Score": [85.78, 85.78, 85.78, 86.19, 86.64, 86.94, 86.94, 87.29, 87.26, 87.19, 87.19, 87.21, 87.33, 87.48, 87.48, 87.48, 87.48, 87.22, 87.22, 87.05, 87.05],
    "Std F1 Score": [2.77, 2.77, 2.77, 2.66, 2.70, 2.43, 2.43, 2.46, 2.38, 2.25, 2.25, 2.15, 2.43, 2.49, 2.49, 2.09, 2.09, 2.24, 2.24, 2.40, 2.40]
}

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# Ensure that numerical values are correctly formatted and converted
df["Mean F1 Score"] = df["Mean F1 Score"].astype(float)
df["Std F1 Score"] = df["Std F1 Score"].astype(float)

# Plot Mean F1 Score with Standard Deviation as error bars (without solid color for std)
plt.figure(figsize=(7, 5), dpi=400)
plt.plot(df["Alpha"], df["Mean F1 Score"], '-o', color='b', label="Mean F1 Score")

# Add error bars using vertical lines without T-bars
plt.vlines(df["Alpha"], df["Mean F1 Score"] - df["Std F1 Score"], df["Mean F1 Score"] + df["Std F1 Score"], color='b', alpha=0.5, label="Std Deviation")

# Highlight the best alpha value (0.75) using a red cross marker
best_alpha = 0.75
best_f1 = df[df["Alpha"] == best_alpha]["Mean F1 Score"].values[0]
plt.scatter(best_alpha, best_f1, color='red', s=100, marker='x', linewidth=2, label="Best Alpha (0.75)")

# Labels and title
plt.xlabel("Alpha")
plt.ylabel("Mean F1 Score")
plt.title("Searching for Optimal Alpha")
plt.legend(loc='lower right')
plt.grid(True, linestyle='dashed')
# plt.tight_layout()
# Save the plot
plt.savefig(f'{output_path}/p22 graph for alpha searching.png', dpi=600)

# Show the plot
plt.show()