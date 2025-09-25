'''
Date: Feb 15, 2024
Task: Searching for optimized alpha value to maximize f1 score
'''

import numpy as np
from scipy.stats import gaussian_kde
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file
import os, pickle
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real

"""====================== Diameter Prior ==================================="""
# Make diameter prior KDE
benign_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_benign_nodule_dia.npy")
malignant_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_malignant_nodule_dia.npy")

kde_benign = gaussian_kde(benign_nodule_sizes)
kde_malignant = gaussian_kde(malignant_nodule_sizes)


# alpha = 0.75

# diameters = [3, 15, 30]  # Example list of diameters for nodules
# model_probs = [0.5, 0.6, 0.9]  # Example list of model probabilities
# # Output: adjusted_probs = [0.35, 0.76, 0.95]

# # Get the list of adjusted probabilities for each nodule
# adjusted_probs = adjust_probability_list(diameters, model_probs, kde_benign, kde_malignant, alpha=0.5)
# print(adjusted_probs)

"""========================= Loading the probability ======================="""
# Loading Deep and Texture Fused Decision
decision_score_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\All Models Pickle Files"
DEEP_Texture_proba = np.load(os.path.join(decision_score_folder, "DEEP_Texture_Fused_proba.npy"))
DEEP_Texture_y = (DEEP_Texture_proba > 0.5).astype(int)
DEEP_Texture_proba = list(DEEP_Texture_proba)
DEEP_Texture_proba = [round(x, 3) for x in DEEP_Texture_proba]

# loading y_true
score_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\All Models Pickle Files"
with open(f'{score_folder}/DEEP_Image_score_with_dia_Eff.pkl', 'rb') as file:
    a_score = pickle.load(file)
    y_true = a_score['y_true_all']
# y_true = np.array(y_true)


diameters_all = np.load(os.path.join(decision_score_folder, "diameters_all.npy"))
diameters_all = list(diameters_all)

for alpha in np.arange(0,1.1,0.05):
    # print(alpha)
    dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=alpha)
    dia_adjusted_proba = np.array(dia_adjusted_proba)
    y_dia_adjusted = ( dia_adjusted_proba >= 0.5).astype(int)
    print(f"F1 Score for alpha = {alpha:.2f}:", f1_score(y_true, y_dia_adjusted))


# Define the optimization function for alpha
def optimize_alpha(alpha):
    Dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=alpha[0])
    Dia_adjusted_proba = np.array(Dia_adjusted_proba)  # Ensure conversion to NumPy array
    y_pred = (Dia_adjusted_proba > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return -f1  # Minimize negative F1-score

# Define the search space for alpha
search_space_alpha = [Real(0.5, 1)]

# Perform Bayesian optimization for alpha
result_alpha = gp_minimize(optimize_alpha, search_space_alpha, n_calls=100, random_state=63)

# Get the optimal alpha
optimal_alpha = result_alpha.x[0]

# Calculate the final Dia_adjusted_proba with the optimized alpha
Dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=optimal_alpha)
Dia_adjusted_proba = np.array(Dia_adjusted_proba)  # Ensure conversion to NumPy array
np.save("Dia_adjusted_proba.npy", Dia_adjusted_proba)

# Final predictions
final_prediction = (Dia_adjusted_proba > 0.5).astype(int)

# print("Optimal Weights:", optimal_weights)
print("Optimal Alpha:", optimal_alpha)
print("Final F1 Score:", f1_score(y_true, final_prediction))

