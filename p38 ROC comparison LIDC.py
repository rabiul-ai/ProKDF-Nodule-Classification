'''
April 2, 2025
Task: Plotting ROC for SPIE-AAPM dataset, paper diagram.
'''

import pickle, os
import numpy as np
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file
import matplotlib.pyplot as plt


output_path = 'Code Outputs/p38 ROC comparison LIDC'

"""==============Step 1:  Loading Scores without dia ======================="""
score_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\All Models Pickle Files\With dia"
all_input_files = os.listdir(score_folder)
input_files = [filename for filename in all_input_files if filename.endswith('.pkl')]

DEEP_GLCM_HOG_LBP_all_proba = []
DEEP_GLCM_HOG_LBP_all_y_true = []

all_scores = []
for idx, filename in enumerate(input_files):
    print(filename)
    with open(f'{score_folder}/{filename}', 'rb') as file:
        a_score = pickle.load(file)
        DEEP_GLCM_HOG_LBP_all_proba.append(a_score['y_proba_all'])
        DEEP_GLCM_HOG_LBP_all_y_true.append(a_score['y_true_all'])
        
DEEP_proba, GLCM_proba, HOG_proba, LBP_proba = DEEP_GLCM_HOG_LBP_all_proba
DEEP_y_true, GLCM_y_true, HOG_y_true, LBP_y_true = DEEP_GLCM_HOG_LBP_all_y_true

# as type of GLCM and HOG y_ture was int64. convert it to int32
GLCM_y_true = list(map(np.int32, GLCM_y_true))
HOG_y_true = list(map(np.int32, HOG_y_true))

'''_________Checking the Sequence of y_tyre for all feature IS EXACTLY SAME__________________'''
if DEEP_y_true == GLCM_y_true == HOG_y_true == LBP_y_true : 
    print('all y true are same')

y_true = DEEP_y_true



"""==============Step 2: Loading Fused Probability ========================"""
Texure_proba = list(np.load(os.path.join(score_folder, "Texture_Fused_proba.npy")))
Texure_y_true = list(np.load(os.path.join(score_folder, "Texture_Fused_y_true.npy")))
DeepText_proba = list(np.load(os.path.join(score_folder, "DEEP_Texture_Fused_proba.npy")))


# Calculate FPR, TPR, AUC for all models
fpr_deep, tpr_deep, _ = roc_curve(DEEP_y_true, DEEP_proba)
auc_deep = auc(fpr_deep, tpr_deep)

fpr_lbp, tpr_lbp, _ = roc_curve(LBP_y_true, LBP_proba)
auc_lbp = auc(fpr_lbp, tpr_lbp)

fpr_hog, tpr_hog, _ = roc_curve(HOG_y_true, HOG_proba)
auc_hog = auc(fpr_hog, tpr_hog)

fpr_glcm, tpr_glcm, _ = roc_curve(GLCM_y_true, GLCM_proba)
auc_glcm = auc(fpr_glcm, tpr_glcm)

fpr_text, tpr_text, _ = roc_curve(DEEP_y_true, Texure_proba)
auc_text = auc(fpr_text, tpr_text)

fpr_DeepText, tpr_DeepText, _ = roc_curve(DEEP_y_true, DeepText_proba)
auc_DeepText = auc(fpr_DeepText, tpr_DeepText)
#%%
# Plot all ROC curves
plt.figure(figsize=(6, 4.5), dpi=500)

plt.plot(fpr_deep, tpr_deep, linestyle=':', color='blue', lw=3, label=f'Deep (AUC = {auc_deep:.2f})')

plt.plot(fpr_lbp, tpr_lbp, linestyle='--', color='darkcyan', lw=1, label=f'LBP (AUC = {auc_lbp:.2f})')
plt.plot(fpr_hog, tpr_hog, linestyle='--', color='deepskyblue', lw=1, label=f'HOG (AUC = {auc_hog:.2f})')
plt.plot(fpr_glcm, tpr_glcm, linestyle='--', color='darkolivegreen', lw=1, label=f'GLCM (AUC = {auc_glcm:.2f})')
plt.plot(fpr_text, tpr_text, lw=2, color='teal', label=f'T Fusion (AUC = {auc_text:.2f})')
plt.plot(fpr_DeepText, tpr_DeepText,  color='red', lw=2, label=f'ProKDF (AUC = {auc_DeepText:.2f})')


# plt.plot(fpr_deep, tpr_deep, label=f'Deep (AUC = {auc_deep:.2f})')
# plt.plot(fpr_lbp, tpr_lbp, label=f'LBP (AUC = {auc_lbp:.2f})')
# plt.plot(fpr_hog, tpr_hog, label=f'HOG (AUC = {auc_hog:.2f})')
# plt.plot(fpr_glcm, tpr_glcm, label=f'GLCM (AUC = {auc_glcm:.2f})')
# plt.plot(fpr_text, tpr_text, label=f'T Fusion (AUC = {auc_text:.2f})')
# plt.plot(fpr_DeepText, tpr_DeepText, label=f'DT Fusion (AUC = {auc_DeepText:.2f})')

# Reference diagonal
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('ROC Curve Comparison Across Feature Types', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='dashed', alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig(f'{output_path}/ROC comparison LIDC.png', dpi=800)

plt.show()

