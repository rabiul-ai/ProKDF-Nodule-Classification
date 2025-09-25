
image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\LBP paper photo making removable"  # Replace with your mounted path

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
import random

plt.rcParams["font.family"] = "Times New Roman"

# Define image folder path
# image_folder = r"C:\Your\Path\To\Images"  # Replace with your actual folder
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(".png")]

# Randomly select 2 images
selected_images = random.sample(image_filenames, 2)

# Define LBP configurations
lbp_settings = [(8, 1), (24, 3), (40, 5)]  # P, R values

# Create figure: 2 rows × 4 columns
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
# fig.suptitle("LBP Feature Maps with Varying P and R", fontsize=12)

for row_idx, filename in enumerate(selected_images):
    image_path = os.path.join(image_folder, filename)
    image_gray = io.imread(image_path, as_gray=True)

    # Column 1: Original
    axes[row_idx, 0].imshow(image_gray)
    if row_idx == 0:
        axes[row_idx, 0].set_title("Nodule")
    axes[row_idx, 0].axis('off')

    # Columns 2–4: LBP variations
    for col_idx, (P, R) in enumerate(lbp_settings, start=1):
        lbp = feature.local_binary_pattern(image_gray, P=P, R=R, method="uniform")
        lbp_normalized = (lbp * 255 / np.max(lbp)).astype(np.uint8)

        axes[row_idx, col_idx].imshow(lbp_normalized)
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(f"LBP (P={P}, R={R})")
        axes[row_idx, col_idx].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
savepath = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p34 LBP comparison paper photo"
output_path = os.path.join(savepath, "LBP_Comparison.png")
plt.savefig(output_path, dpi=500)
plt.show()
