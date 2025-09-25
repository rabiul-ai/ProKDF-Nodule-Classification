


import matplotlib.pyplot as plt
import numpy as np

output_path = 'Code Outputs/p36 F1 score comparison paper'

# Use Times New Roman for all text
plt.rcParams["font.family"] = "Times New Roman"

# Define model names and F1-scores with standard deviation
models = ["Deep", "LBP", "HOG", "GLCM", "T Fused", "DT Fused"]

# F1 scores and standard deviations for LIDC-IDRI
lidc_f1 = [87.48, 82.73, 82.01, 82.48, 83.19, 87.48]
lidc_std = [2.09, 1.62, 2.02, 1.66, 1.09, 2.09]

# F1 scores and standard deviations for SPIE-AAPM
spie_f1 = [72.55, 71.72, 59.76, 68.25, 74.69, 74.69]
spie_std = [4.32, 6.86, 17.41, 4.26, 7.62, 7.62]

# Bar width and positions
x = np.arange(len(models))
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(6, 3), dpi=500)
bars1 = ax.bar(x - width/2, lidc_f1, width, yerr=lidc_std, label='LIDC-IDRI', color='skyblue', error_kw=dict(capsize=0, capthick=0))
bars2 = ax.bar(x + width/2, spie_f1, width, yerr=spie_std, label='SPIE-AAPM', color='salmon', error_kw=dict(capsize=0, capthick=0))

# Add value labels with mean ± std format above error bars
offset = 3
for bar, mean, std in zip(bars1, lidc_f1, lidc_std):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + std + offset, f'{mean:.2f}±{std:.2f}', 
            ha='center', va='bottom', fontsize=8, rotation=90)

for bar, mean, std in zip(bars2, spie_f1, spie_std):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + std + offset, f'{mean:.2f}±{std:.2f}', 
            ha='center', va='bottom', fontsize=8, rotation=90)

# Labels and formatting
ax.set_ylabel('F1-score (%)')
ax.set_xticks(x)
ax.set_xticklabels(models)
# Add 'ProKDF' text below 'DT Fused' bar
dt_fused_index = models.index("DT Fused")
ax.text(dt_fused_index, 42, 'ProKDF', ha='center', va='top')
ax.set_ylim(50, 110)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=2)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_path}/F1 score comparison.png', dpi=600)
plt.show()

