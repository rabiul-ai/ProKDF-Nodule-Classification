import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

# Set Times New Roman font globally
mpl.rcParams['font.family'] = 'Times New Roman'

output_path = 'Code Outputs/p22 graph for alpha searching paper TNR font'

# Define the data
data_dict = {
    "Alpha": np.linspace(0, 1, 21),
    "Mean F1 Score": [85.78, 85.78, 85.78, 86.19, 86.64, 86.94, 86.94, 87.29, 87.26, 87.19, 87.19, 87.21, 87.33, 87.48, 87.48, 87.48, 87.48, 87.22, 87.22, 87.05, 87.05],
    "Std F1 Score": [2.77, 2.77, 2.77, 2.66, 2.70, 2.43, 2.43, 2.46, 2.38, 2.25, 2.25, 2.15, 2.43, 2.49, 2.49, 2.09, 2.09, 2.24, 2.24, 2.40, 2.40]
}

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# Ensure numeric format
df["Mean F1 Score"] = df["Mean F1 Score"].astype(float)
df["Std F1 Score"] = df["Std F1 Score"].astype(float)

# Plotting
plt.figure(figsize=(7, 4), dpi=400)
plt.plot(df["Alpha"], df["Mean F1 Score"], '-o', color='b', label="Mean F1-score")
plt.vlines(df["Alpha"], df["Mean F1 Score"] - df["Std F1 Score"], df["Mean F1 Score"] + df["Std F1 Score"], color='b', alpha=0.5, label="Std. deviation")

# Highlight best alpha
best_alpha = 0.75
best_f1 = df[df["Alpha"] == best_alpha]["Mean F1 Score"].values[0]
plt.scatter(best_alpha, best_f1, color='red', s=100, marker='x', linewidth=2, label=r"Optimal $\alpha$ = 0.75")


# Labels and title in Times New Roman
plt.xlabel(r"$\alpha$ values")
plt.ylabel("F1-score (mean ± std.)")
# plt.title("Searching for Optimal Alpha")
plt.legend(loc='lower right')
plt.grid(True, linestyle='dashed')
plt.tight_layout()

# Save and show
plt.savefig(f'{output_path}/p22 graph for alpha searching.png', dpi=600)
plt.show()
