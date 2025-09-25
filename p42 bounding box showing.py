import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from matplotlib.patches import Rectangle
import random

# Set seed for reproducibility
random.seed(61)

# Folder containing the mask images
folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules Predicted Mask"

# List all mask files (assuming .png; change extension if needed)
all_files = [f for f in os.listdir(folder) if f.endswith('.png')]

# Randomly select 40 images
random_files = random.sample(all_files, 40)

# Create figure for plotting
fig, axes = plt.subplots(8, 5, figsize=(20, 32))
axes = axes.ravel()  # flatten axes for easy iteration

for i, file in enumerate(random_files):
    # Load mask image
    mask = imread(os.path.join(folder, file))
    
    # If RGB, convert to grayscale by taking first channel
    if mask.ndim == 3:
        mask = mask[..., 0]
    
    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)
    
    # Label connected components
    labeled_mask = label(mask)
    
    # Get properties of regions
    regions = regionprops(labeled_mask)
    
    if regions:
        # Largest region
        largest_region = max(regions, key=lambda x: x.area)
        minr, minc, maxr, maxc = largest_region.bbox
        diameter_pixels = np.linalg.norm([maxr - minr, maxc - minc])
        diameter_mm = diameter_pixels * 0.5  # convert to mm
    else:
        # No region found
        minr = minc = maxr = maxc = 0
        diameter_mm = 0
    
    # Plot mask
    axes[i].imshow(mask, cmap='gray')
    
    # Plot bounding box in red
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                     edgecolor='red', facecolor='none', linewidth=2)
    axes[i].add_patch(rect)
    
    # Add diameter text inside the box (top-left corner)
    axes[i].text(minc, max(0, minr-2), f'{diameter_mm:.2f} mm', 
                 color='yellow', fontsize=20, weight='bold')
    
    axes[i].axis('off')
    axes[i].set_title(f'Nodule {i+1}', fontsize=15)

plt.tight_layout()

save_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p42 bounding box showing\largest_bbox_plot.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # high-resolution PNG
plt.show()
