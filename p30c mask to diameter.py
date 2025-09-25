import cv2
import numpy as np
from skimage.measure import label, regionprops

image = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules Predicted Mask\nid_10_s_4_dia_26_p_CT-Training-LC009_n_1_c_Y.png"
# Load the mask image (binary image)
mask = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.uint8)  # Convert to binary mask

# Label connected components (regions)
labeled_mask = label(mask)

# Get properties of the largest region (nodule)
regions = regionprops(labeled_mask)

# Assuming the largest region is the nodule
largest_region = max(regions, key=lambda x: x.area)

# Get the coordinates of the bounding box of the nodule
minr, minc, maxr, maxc = largest_region.bbox

# Calculate the diameter (distance between furthest points along the region)
diameter_pixels = np.linalg.norm([maxr - minr, maxc - minc])

# Convert to mm (scale factor = 0.5 mm/pixel)
diameter_mm = diameter_pixels * 0.5

print(f"Predicted nodule diameter: {diameter_mm:.2f} mm")
