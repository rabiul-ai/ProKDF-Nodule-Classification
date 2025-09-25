'''
=============================== ABOUT CODING ==================================
Date: Feb 19, 2025. Copy of p17 9v searching for best model.py________________________________________
Task:  GRID SEARCH FOR ALPHA, only for efficientNet, with majority voting
fusion_style = 'majority', 'average'
________________________________________________________

'''

code_no = 'p21' # just checking a simple 2D CNN based model accuracy
output_path = 'Code Outputs/p21 grids search alpha'
malignancy_3 ='ignored' #'benign' 'malignant'
fusion_style = 'majority'

"""============================= Importing Libraries ====================== """
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import os
import csv
from scipy.stats import gaussian_kde
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file
from sklearn.model_selection import KFold

alpha_values = np.round(np.linspace(0, 1, 21), 2).tolist()
# alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]



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



"""======================= Loading Data  ==================================="""
DEEP_prediction_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p17 majority voting Res Eff\DEEP_Image_score_Eff.pkl"
with open(DEEP_prediction_path, 'rb') as file:
    DEEP_prediction = pickle.load(file)


image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images"
# update_1: Using 32 sized image can it improve result or not. Let's check it. Finding: Result slight decreased.
# image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules\Images"

nodule_to_images = {}

for filename in os.listdir(image_folder):    
    # _________________ FILTERING OUT NODULES OF malignancy = 3 _______________
    if malignancy_3 == 'ignored':
        if '_c_U' not in filename:
            parts = filename.split('_')
            nid = parts[1]  # Extract nodule ID
            if nid not in nodule_to_images:
                nodule_to_images[nid] = []
            nodule_to_images[nid].append(filename)
        
    else: # malignancy_3 == 'benign' or 'malignant'
        parts = filename.split('_')
        nid = parts[1]  # Extract nodule ID
        if nid not in nodule_to_images:
            nodule_to_images[nid] = []
        nodule_to_images[nid].append(filename)

# Counting_____________________________________________________________________
print(f'Number of total nodules = {len(nodule_to_images)}')
n_benign, n_malignant = 0, 0
for key, value in nodule_to_images.items():
    if '_c_Y' in value[0]:
        n_malignant += 1 
    else:
        n_benign += 1 
print(f'Bening nodules = {n_benign}, Malignant nodules = {n_malignant}')

# Sorting______________________________________________________________________
nodule_ids = sorted(nodule_to_images.keys())



"""============================== Training Models ========================="""



n_fold = 7
kf = KFold(n_splits=n_fold, shuffle=True, random_state=63)
folds = list(kf.split(nodule_ids))

# diameters_all = []
""" ==================== Iterating through models ============================"""
for alpha in alpha_values:
    
    results = []
    accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities = [], [], [], [], [], [], []
    scores = {}
    
    '''_________________Opening New Dictionary for Saving Decision______________'''
    DEEP_Image_score_with_dia = {}
    keys = ['y_true_fold','y_pred_fold', 'y_proba_fold', 'y_true_all','y_pred_all', 'y_proba_all']
    for key in keys:
        DEEP_Image_score_with_dia[key] = []
    
    for fold, (train_idx, test_idx) in enumerate(folds):
    # for fold in range(7):
        y_test = DEEP_prediction['y_true_fold'][fold]
        y_pred_prob = DEEP_prediction['y_proba_fold'][fold]
        y_pred = DEEP_prediction['y_pred_fold'][fold]

        train_nodules = [nodule_ids[i] for i in train_idx]  # list, len 808, ['1', '10', ...]
        test_nodules = [nodule_ids[i] for i in test_idx]
        
        train_images = [img for nid in train_nodules for img in nodule_to_images[nid]]   # list of imagenames, len = 808*9 = 7272
        test_images = [img for nid in test_nodules for img in nodule_to_images[nid]]
        
        # Getting diameter information of test nodules_____________________________
        diameters_test_images = [image_name.split('_')[5] for image_name in test_images]
        diameters_test_nodules = diameters_test_images[0::9] # every nineth element
        diameters_test_nodules = [int(dia) for dia in diameters_test_nodules]
        
        '''____________________ ADJUST WITH DIAMETER_________________________'''
        # print(y_pred_prob)
        y_pred_prob = np.round(y_pred_prob, 2) # making 2 decimal point
        model_probs = list(y_pred_prob)
        y_pred_prob_adjusted = adjust_probability_list(diameters_test_nodules, model_probs, kde_benign, kde_malignant, alpha=alpha)
        y_pred_prob_adjusted = np.array(y_pred_prob_adjusted)
        y_pred_adjusted = ( y_pred_prob_adjusted >= 0.5).astype(int)


        
        '''_______________________ Saving Decision for Fusion with Dia Adjustment__________'''
        DEEP_Image_score_with_dia['y_true_fold'].append(list(y_test))
        DEEP_Image_score_with_dia['y_pred_fold'].append(list(y_pred_adjusted))
        DEEP_Image_score_with_dia['y_proba_fold'].append(list(y_pred_prob_adjusted))

        DEEP_Image_score_with_dia['y_true_all'].extend(list(y_test))
        DEEP_Image_score_with_dia['y_pred_all'].extend(list(y_pred_adjusted))
        DEEP_Image_score_with_dia['y_proba_all'].extend(list(y_pred_prob_adjusted))
        
        
        # '''______________________Evaluating Performance_____________________'''
        # performance_metrices2 = evaluate_and_print_performance(fold, y_test, y_pred, y_pred_prob, accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2)
        # accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2 = performance_metrices2  
         
        '''______________ After Diameter Adjustment ________________________'''
        performance_metrices = evaluate_and_print_performance(fold, y_test, y_pred_adjusted, y_pred_prob_adjusted, accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores)
        accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities, scores = performance_metrices  
            
        
        

    '''________________Mean Scores of All Folds_____________________________'''
    results, scores = get_mean_performance_of_all_folds(performance_metrices, model_name = f'Alpha = {alpha}')
    save_scores_in_txt_file(output_path, code_no, scores, alpha, model_name = f'Alpha = {alpha}', dia = 'Yes')
       
                
    # '''________________Mean Scores of All Folds_____________________________'''
    # results2, scores2 = get_mean_performance_of_all_folds(performance_metrices2, model_name = model_name)
    # save_scores_in_txt_file(output_path, code_no, scores2, alpha, model_name, dia = 'No')

    """=============================Saving Results=============================="""
    # Convert results to a DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(f'{output_path}/{code_no} Alpha_{alpha}_model_comparison.xlsx', index=False)
    print(f'{output_path}/{code_no} Model comparison saved to model_comparison.xlsx')


    """=================== Saving Decision ================================="""
    import pickle

    files_to_be_saved = [DEEP_Image_score_with_dia]
    file_names = ['DEEP_Image_score_with_dia']
    # tag = model_name[:3]
    for variable, filename in zip(files_to_be_saved, file_names):
        # saving pickle file_____________________
        with open(f'{output_path}/{filename}_Alpha_{alpha}.pkl', 'wb') as file:
            pickle.dump(variable, file)

