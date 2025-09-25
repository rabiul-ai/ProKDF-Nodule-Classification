'''
Date: Feb 16, 2025
Task: Getting 7 fold cross validation score of final result get in p15
Input: Texture and Deep Fused Decision
'''

import pickle
import numpy as np
from os.path import join as opj
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file
from sklearn.model_selection import KFold

code_no = 'p16' # just checking a simple 2D CNN based model accuracy
output_path = 'Code Outputs/p16 cross validation score'

#________________ Loading Data________________________
score_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\All Models Pickle Files\With dia"
filename = "Texture_Fused_proba.npy"
y_proba_all = np.load(opj(score_folder, filename))
y_pred_all = (y_proba_all > 0.5).astype(int)
with open(f'{score_folder}/DEEP_Image_score_with_dia_Eff.pkl', 'rb') as file:
    a_score = pickle.load(file)
    y_true_all = a_score['y_true_all']
# y_true_all = np.array(y_true_all)

y_proba_all = list(y_proba_all)
y_pred_all = list(y_pred_all)

# _______________making fold wise________________________
# among 7 fold first 5 folds elements was 135 and rest 134

def split_list(lst):
    chunk_sizes = [135] * 5 + [134] * 2
    result = []
    index = 0
    for size in chunk_sizes:
        result.append(lst[index : index + size])
        index += size
    return result

y_true_fold = split_list(y_true_all)
y_proba_fold = split_list(y_proba_all)
y_pred_fold = split_list(y_pred_all)


results = []
accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities = [], [], [], [], [], [], []
scores = {}

for n_fold in range(7):
    y_true = np.array(y_true_fold[n_fold])
    y_proba = np.array(y_proba_fold[n_fold])
    y_pred = np.array(y_pred_fold[n_fold])
    
    performance_metrices = evaluate_and_print_performance(n_fold, y_true, y_pred, y_proba, accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores)
    accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities, scores = performance_metrices  
    # break

'''________________Mean Scores of All Folds_____________________________'''
results, scores = get_mean_performance_of_all_folds(performance_metrices, model_name = 'Texture Fused')
save_scores_in_txt_file(output_path, code_no, scores, alpha=0.75, model_name = 'Texture Fused', dia = 'Yes')

