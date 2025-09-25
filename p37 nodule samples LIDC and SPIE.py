import os
import random
import re
import matplotlib.pyplot as plt
from skimage import io

# -------------------- Settings -------------------- #
random_seed = 101 # Change to any number to get different samples
random.seed(random_seed)
plt.rcParams['font.family'] = 'Times New Roman'
output_path = 'Code Outputs/p37 nodule samples LIDC and SPIE'

# -------------------- Dataset Paths -------------------- #
lidc_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images"
spie_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules"

# -------------------- Load LIDC -------------------- #
lidc_all = os.listdir(lidc_path)
lidc_benign = [f for f in lidc_all if '_m_1' in f or '_m_2' in f]
lidc_malignant = [f for f in lidc_all if '_m_4' in f or '_m_5' in f]
lidc_benign_samples = random.sample(lidc_benign, min(7, len(lidc_benign)))
lidc_malignant_samples = random.sample(lidc_malignant, min(7, len(lidc_malignant)))

# -------------------- Load SPIE -------------------- #
spie_all = os.listdir(spie_path)
spie_benign = [f for f in spie_all if '_c_N' in f]
spie_malignant = [f for f in spie_all if '_c_Y' in f]
spie_benign_samples = random.sample(spie_benign, min(7, len(spie_benign)))
spie_malignant_samples = random.sample(spie_malignant, min(7, len(spie_malignant)))

# -------------------- Utility Function -------------------- #
def extract_m_value(filename):
    match = re.search(r'm_(\d+)', filename)
    return match.group(1) if match else ""

def plot_dataset(fig, axes, benign_list, malignant_list, base_path, label_text):
    num_cols = max(len(benign_list), len(malignant_list))
    
    for i in range(num_cols):
        # Benign
        if i < len(benign_list):
            img_path = os.path.join(base_path, benign_list[i])
            img = io.imread(img_path, as_gray=True)
            m_val = extract_m_value(benign_list[i])
            axes[0, i].imshow(img, cmap='gray')
            # axes[0, i].set_title(f"m = {m_val}" if m_val else "", fontsize=12)
        axes[0, i].axis('off')
        
        # Malignant
        if i < len(malignant_list):
            img_path = os.path.join(base_path, malignant_list[i])
            img = io.imread(img_path, as_gray=True)
            m_val = extract_m_value(malignant_list[i])
            axes[1, i].imshow(img, cmap='gray')
            # axes[1, i].set_title(f"m = {m_val}" if m_val else "", fontsize=12)
        axes[1, i].axis('off')
    
    # Centered dataset label
    x_center = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
    y_top = axes[0, 0].get_position().y1 + 0.02

    if label_text == 'LIDC-IDRI':
    # Manually set positions for LIDC
        fig.text(0.06, 0.83, 'Benign', va='center', ha='center', fontsize=15, rotation='vertical')
        fig.text(0.06, 0.61, 'Malignant', va='center', ha='center', fontsize=15, rotation='vertical')
        fig.text(0.03, 0.71, 'LIDC-IDRI', va='center', ha='center', fontsize=15, rotation='vertical')

    elif label_text == 'SPIE-AAPM':
    # Manually set positions for SPIE
        fig.text(0.06, 0.38, 'Benign', va='center', ha='center', fontsize=15, rotation='vertical')
        fig.text(0.06, 0.16, 'Malignant', va='center', ha='center', fontsize=15, rotation='vertical')
        fig.text(0.03, 0.27, 'SPIE-AAPM', va='center', ha='center', fontsize=15, rotation='vertical')

# -------------------- Create Final Plot -------------------- #
total_cols = max(
    max(len(lidc_benign_samples), len(lidc_malignant_samples)),
    max(len(spie_benign_samples), len(spie_malignant_samples))
)

fig, axes_all = plt.subplots(4, total_cols, figsize=(1.5*total_cols, 6))
axes_all = axes_all.reshape(4, total_cols)

# Top half: LIDC-IDRI
plot_dataset(fig, axes_all[0:2, :], lidc_benign_samples, lidc_malignant_samples, lidc_path, 'LIDC-IDRI')

# Bottom half: SPIE-AAPM
plot_dataset(fig, axes_all[2:4, :], spie_benign_samples, spie_malignant_samples, spie_path, 'SPIE-AAPM')


# Adjust subplot spacing manually
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, wspace=0.005, hspace=0.06)

# plt.tight_layout(rect=[0.08, 0, 1, 1])
os.makedirs(output_path, exist_ok=True)
plt.savefig(f'{output_path}/lidc_spie_examples {random_seed}.png', dpi=600)
plt.show()
