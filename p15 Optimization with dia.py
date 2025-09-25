'''
Date: February 16, 2025
Task: Finding weights for the scores with dia
    Copy of c14
    Combination of code p11, p12 and p13
    for maximizing f1 score.
'''

import pickle, os
import numpy as np
import numpy as np
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import gaussian_kde
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file


"""==============Step 1: Searching Weights for Fusion of LBP, HOG and GLCM ============"""
"""______________________ Loading Scores without dia____________________________"""
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



"""__________________________ Bayesian Optimization ______________________"""
# Stack the probability scores for GLCM, HOG, and LBP
prob_scores = np.array([GLCM_proba, HOG_proba, LBP_proba])

# Define the optimization function to maximize the F1-score
def optimize_weights(weights):
    weights = np.array(weights)
    weights /= weights.sum()  # Ensure weights sum to 1
    weighted_avg_prob = np.average(prob_scores, axis=0, weights=weights)
    y_pred = (weighted_avg_prob > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return -f1  # Minimize negative F1-score

# Define the search space for weights
search_space = [Real(0, 1), Real(0, 1), Real(0, 1)]

# Perform Bayesian optimization
result = gp_minimize(optimize_weights, search_space, n_calls=100, random_state=0)

# Get the optimal weights
optimal_weights = result.x
optimal_weights = np.array(optimal_weights)
optimal_weights /= optimal_weights.sum()  # Ensure weights sum to 1

# Calculate the weighted average probabilities with optimal weights
weighted_avg_prob = np.average(prob_scores, axis=0, weights=optimal_weights)
final_prediction = (weighted_avg_prob > 0.5).astype(int)
np.save(os.path.join(score_folder, "Texture_Fused_proba.npy"), weighted_avg_prob)
np.save(os.path.join(score_folder, "Texture_Fused_y_true.npy"), final_prediction)

print("Optimal Weights:", optimal_weights)
print("Final F1 Score:", f1_score(y_true, final_prediction))



# Calculate F1-score using equal weights (1/3, 1/3, 1/3)
equal_weights = np.array([1/3, 1/3, 1/3])
equal_weighted_avg_prob = np.average(prob_scores, axis=0, weights=equal_weights)
equal_final_prediction = (equal_weighted_avg_prob > 0.5).astype(int)
equal_f1_score = f1_score(y_true, equal_final_prediction)
print("F1 Score with Equal Weights (1/3, 1/3, 1/3):", equal_f1_score)


#%%


"""============ Step 2: Searching Weights for Deep and Texture Fusion ======="""
"""========================== Optimization ================================="""
# Loading texture fused information
Texture_proba = np.load(os.path.join(score_folder, "Texture_Fused_proba.npy"))
Texture_proba = list(Texture_proba)

# Stack the probability scores for DEEP and Texture models
prob_scores = np.array([DEEP_proba, Texture_proba])

# Define the optimization function to maximize the F1-score
def optimize_weights(weights):
    weights = np.array(weights)
    weights /= weights.sum()  # Ensure weights sum to 1
    weighted_avg_prob = np.average(prob_scores, axis=0, weights=weights)
    y_pred = (weighted_avg_prob > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return -f1  # Minimize negative F1-score

# Define the search space for weights
search_space = [Real(0, 1), Real(0, 1)]

# Perform Bayesian optimization
result = gp_minimize(optimize_weights, search_space, n_calls=100, random_state=0)

# Get the optimal weights
optimal_weights = result.x
optimal_weights = np.array(optimal_weights)
optimal_weights /= optimal_weights.sum()  # Ensure weights sum to 1

# Calculate the weighted average probabilities with optimal weights
weighted_avg_prob = np.average(prob_scores, axis=0, weights=optimal_weights)
final_prediction = (weighted_avg_prob > 0.5).astype(int)
np.save(os.path.join(score_folder, "DEEP_Texture_Fused_proba.npy"), weighted_avg_prob)

print("Optimal Weights:", optimal_weights)
print("Final F1 Score:", f1_score(y_true, final_prediction))



#%%

"""================== Step 3: Searching for Alpha value ===================="""
"""====================== Diameter Prior ==================================="""
# Make diameter prior KDE
benign_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_benign_nodule_dia.npy")
malignant_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_malignant_nodule_dia.npy")

kde_benign = gaussian_kde(benign_nodule_sizes)
kde_malignant = gaussian_kde(malignant_nodule_sizes)


'''_______________________________ Loading Data ___________________________'''
# Loading Deep and Texture Fused Decision
DEEP_Texture_proba = np.load(os.path.join(score_folder, "DEEP_Texture_Fused_proba.npy"))
DEEP_Texture_y = (DEEP_Texture_proba > 0.5).astype(int)
DEEP_Texture_proba = list(DEEP_Texture_proba)
DEEP_Texture_proba = [round(x, 3) for x in DEEP_Texture_proba]

diameters_all = np.load(os.path.join(score_folder, "diameters_all.npy"))
diameters_all = list(diameters_all)

'''________________________ Optimization __________________________________'''

# Define the optimization function for alpha
def optimize_alpha(alpha):
    Dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=alpha[0])
    Dia_adjusted_proba = np.array(Dia_adjusted_proba)  # Ensure conversion to NumPy array
    y_pred = (Dia_adjusted_proba > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    return -f1  # Minimize negative F1-score

# Define the search space for alpha
search_space_alpha = [Real(0, 1)]

# Perform Bayesian optimization for alpha
result_alpha = gp_minimize(optimize_alpha, search_space_alpha, n_calls=100, random_state=63)

# Get the optimal alpha
optimal_alpha = result_alpha.x[0]

# Calculate the final Dia_adjusted_proba with the optimized alpha
Dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=optimal_alpha)
Dia_adjusted_proba = np.array(Dia_adjusted_proba)  # Ensure conversion to NumPy array
np.save("Alpha_adjusted_proba_Final.npy", Dia_adjusted_proba)

# Final predictions
final_prediction = (Dia_adjusted_proba > 0.5).astype(int)

# print("Optimal Weights:", optimal_weights)
print("Optimal Alpha:", optimal_alpha)
print("Final F1 Score:", f1_score(y_true, final_prediction))


# # Grid Search, jsut for checking more
# for alpha in np.arange(0,1.1,0.05):
#     # print(alpha)
#     dia_adjusted_proba = adjust_probability_list(diameters_all, DEEP_Texture_proba, kde_benign, kde_malignant, alpha=alpha)
#     dia_adjusted_proba = np.array(dia_adjusted_proba)
#     y_dia_adjusted = ( dia_adjusted_proba >= 0.5).astype(int)
#     print(f"F1 Score for alpha = {alpha:.2f}:", f1_score(y_true, y_dia_adjusted))
