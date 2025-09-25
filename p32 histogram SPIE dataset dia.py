"""
Task1: Nodule Size Histogram Plotting
Dataset: SPIE-AAPM
Date: March 6, 2024
"""


"""=============================== Importing Libraries ====================="""
import pylidc as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from statistics import median_high
import numpy as np
import pandas as pd


code_output = 'Code Outputs/p32 histogram SPIE dataset dia'
image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules"

nodule_to_images = {}

for filename in os.listdir(image_folder):  
    parts = filename.split('_')
    nid = parts[1]  # Extract nodule ID
    if nid not in nodule_to_images:
        nodule_to_images[nid] = []
    nodule_to_images[nid].append(filename)
    

# Counting_____________________________________________________________________
benign_nodule_sizes, malignant_nodule_sizes = [], []


print(f'Number of total nodules = {len(nodule_to_images)}')
n_benign, n_malignant = 0, 0
for key, value in nodule_to_images.items():
    dia = value[0].split('_')[5]
    dia = int(dia)
    if '_c_Y' in value[0]:
        n_malignant += 1 
        malignant_nodule_sizes.append(dia)
    else:
        n_benign += 1 
        benign_nodule_sizes.append(dia)
print(f'Bening nodules = {n_benign}, Malignant nodules = {n_malignant}')


n_bins = 30

"""=================== Plotting Histogram Benign vs Malignant =============="""
# ___________________Make a common Bin_____________________        
# Determine the common bin range by finding the min and max of all data
min_bin = min(min(benign_nodule_sizes), min(malignant_nodule_sizes))
max_bin = max(max(benign_nodule_sizes), max(malignant_nodule_sizes))

# Create common bins
common_bins = np.linspace(min_bin, max_bin, n_bins)  # 30 bins between min_bin and max_bin

# Plotting the histogram of nodule sizes
plt.figure(figsize=(6, 3.5), dpi=500)
# plt.hist(nodule_sizes, bins=common_bins,  color='skyblue', edgecolor='blue', alpha=0.2)
plt.hist(benign_nodule_sizes, bins=common_bins,  hatch='////', color='green', linewidth=1, edgecolor='green', alpha=0.35)
plt.hist(malignant_nodule_sizes, bins=common_bins, hatch='..',  color='red', linewidth=1, edgecolor='red', alpha=0.35)
# plt.title("Histogram of Nodule Sizes (Diameter)")
plt.xlabel("Nodule Size (mm)")
plt.ylabel("Nodule Count")
plt.legend(['Benign (# 815)', 'Malignant (# 570)'])
plt.tight_layout()
plt.savefig(f'{code_output}/nodule histogram.png')
plt.show()



"""======================== Extra: Histogram and KDE plotting =============="""
import seaborn as sns
common_bins = np.linspace(min_bin, max_bin, n_bins)
sns.set(style="darkgrid")
plt.figure(figsize=(6, 3.5), dpi=400)
sns.histplot(benign_nodule_sizes, bins=common_bins, color='green', kde=True, label='Benign Nodules', alpha=0.3)
sns.histplot(malignant_nodule_sizes, bins=common_bins, color='red', kde=True, label='Malignant Nodules', alpha=0.3)
plt.xlabel('Nodule Size (mm)')
plt.ylabel('Nodule Count')
plt.legend()
plt.tight_layout()
plt.savefig(f'{code_output}/nodule histogram with KDE.png')
plt.show()



import matplotlib.font_manager as fm
# Force Times New Roman from system fonts
font_path = fm.findfont("Times New Roman")
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

# Apply font globally for matplotlib and seaborn
plt.rcParams['font.family'] = font_name
# sns.set(style="darkgrid", font=font_name)
# sns.set(style="whitegrid", font=font_name)
sns.set(style="white", font=font_name)



import seaborn as sns
common_bins = np.linspace(min_bin, max_bin, n_bins)
sns.set(style="white", font=font_name)
plt.figure(figsize=(6, 3.5), dpi=400)
sns.histplot(benign_nodule_sizes, bins=common_bins, color='green', kde=True, label='Benign', alpha=0.3)
sns.histplot(malignant_nodule_sizes, bins=common_bins, color='red', kde=True, label='Malignant', alpha=0.3)
plt.xlabel('Estimated diameter (mm)', fontproperties=font_prop)
plt.ylabel('Nodule count', fontproperties=font_prop)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=1)
sns.despine()

plt.tight_layout()
plt.savefig(f'{code_output}/nodule histogram with KDE.png')
plt.show()


"""======================= Saving the Nodule Size Lists==================== """
np.save(f'{code_output}/all_SPIE_benign_nodule_dia.npy', benign_nodule_sizes)
np.save(f'{code_output}/all_SPIE_malignant_nodule_dia.npy', malignant_nodule_sizes)


