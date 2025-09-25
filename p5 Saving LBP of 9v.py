"""Making LBP images of 9V"""

import os
import numpy as np
from skimage import io, feature, color
import matplotlib.pyplot as plt
# Define paths
image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images"
# image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules"
# predicted_mask_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Predicted Masks"

output_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\LBP Images"
# output_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\SPIE AAPM LBP Images"
# image_folder = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Code 5 Final Try\Nodule and Mask Images\Nodules testing code"
# mask_folder = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Code 5 Final Try\Nodule and Mask Images\Masks testing code"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

threshold = 0.5

a = 0
aa = os.listdir(image_folder)

# Process each image
for filename in os.listdir(image_folder)[a:a+1]:
    if filename.endswith(".png"):  # adjust according to your image format
        # Load image and corresponding mask
        ap = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images\nid_1_s_1_dia_33_p_0001_n_1_m_5_c_Y.png"
        image_path = os.path.join(image_folder, ap)
        # mask_path = os.path.join(predicted_mask_folder, filename)
        
        
        gray_image = io.imread(image_path, as_gray=True)  
        plt.imshow(gray_image)
        plt.show()
        # mask = io.imread(mask_path, as_gray=True)  # Assuming mask is a binary image
        # mask_bool = mask > threshold
        
        # Compute LBP
        lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method="uniform")
        plt.imshow(lbp)
        plt.show()
        
        # masked_lbp_amplified = lbp*mask_bool
        # masked_lbp_amplified = (masked_lbp_amplified * 255/np.max(masked_lbp_amplified.ravel())).astype(np.uint8)
        # io.imshow(masked_lbp_amplified)
        
        # Save the masked LBP image
        output_path = os.path.join(output_folder, filename)
        lbp = (lbp * 255/np.max(lbp.ravel())).astype(np.uint8)
        # io.imsave(output_path, lbp)
        # break


# io.imshow(masked_lbp_amplified)
