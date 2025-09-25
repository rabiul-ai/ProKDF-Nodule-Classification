'''
=============================== ABOUT CODING ==================================
Date: Feb 8, 2025. Copy of p6 ________________________________________
Task: Checking Performance of HOG Features, 9views. ResNet & EfficeintNet

fusion_style = 'majority', 'average'
________________________________________________________

'''

code_no = 'p8' # just checking a simple 2D CNN based model accuracy
output_path = 'Code Outputs/p8 HOG_feature LR SVM kNN'
malignancy_3 ='ignored' #'benign' 'malignant'
fusion_style = 'average'

"""============================= Importing Libraries ====================== """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
import os
from scipy.stats import gaussian_kde
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Training parameters
n_fold = 7
n_batch = 16
n_epoch = 50
n_patience = 10

# Define the models
model_names = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'kNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}


"""====================== Diameter Prior ==================================="""
# Make diameter prior KDE
benign_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_benign_nodule_dia.npy")
malignant_nodule_sizes = np.load(r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc1 output histogram\all_2625_malignant_nodule_dia.npy")

kde_benign = gaussian_kde(benign_nodule_sizes)
kde_malignant = gaussian_kde(malignant_nodule_sizes)


alpha = 0.75

# diameters = [3, 15, 30]  # Example list of diameters for nodules
# model_probs = [0.5, 0.6, 0.9]  # Example list of model probabilities
# # Output: adjusted_probs = [0.35, 0.76, 0.95]

# # Get the list of adjusted probabilities for each nodule
# adjusted_probs = adjust_probability_list(diameters, model_probs, kde_benign, kde_malignant, alpha=0.5)
# print(adjusted_probs)



"""======================= Loading Data  ==================================="""

'''________________________ Loading HOG Features____________________________'''
HOG_feature_filepath = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p7 extracting textures features 9v\HOG_features.xlsx"

HOG_features = pd.read_excel(HOG_feature_filepath)
X_value = HOG_features.iloc[:, 1:-3].values
y_d1 = HOG_features['d1_label'].values 
y_d2 = HOG_features['d2_label'].values 
y_d3 = HOG_features['d3_label'].values


# saving a smaller version_______________________________________
# HOG_features_1000 = HOG_features.head(1000)
# HOG_features_1000.to_excel("HOG_features_1000.xlsx", index=False)



'''_____________Loading image filenames grouped by NODULEs____________________'''
image_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images"
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


def load_selected_HOG_features_and_labels(image_list, HOG_features):
    # Update 2, as we strictly need to maintain sorting as order of image list. otherwise in decision fusion it will create problem.
    HOG_features_sampled = HOG_features[HOG_features['nodule_id'].isin(image_list)]
    HOG_features_sampled['nodule_id'] = pd.Categorical(HOG_features_sampled['nodule_id'], categories=image_list, ordered=True)
    HOG_features_sorted =  HOG_features_sampled.sort_values('nodule_id')
    X = HOG_features_sorted.iloc[:, 1:-3].values
    y = HOG_features_sorted['d1_label'].values 
    return np.array(X), np.array(y)


'''_______________________ Function for nodule based prediction_____________'''
def predict_nodule_based(model, test_images, image_folder, nodule_to_images):
    nodule_predictions = {}
    for nid, image_files in nodule_to_images.items():
        if nid in test_images:
            image_probs = []
            for img_file in image_files:
                img_path = os.path.join(image_folder, img_file)
                img = load_img(img_path)
                img_array = np.uint8(img_to_array(img)) # img = load_img(img_path, target_size=(224, 224)) # img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prob = model.predict(img_array)[0][0]
                image_probs.append(float(prob))  # Convert to Python float
            nodule_predictions[nid] = float(np.mean(image_probs))  # Convert mean to float
    return nodule_predictions



"""============================== Training Models ========================="""
results = []
accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities = [], [], [], [], [], [], []
scores = {}


# variables for saving scores without diameter adjustment
results2 = []
accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2, sensitivities2, specificities2 = [], [], [], [], [], [], []
scores2 = {}


kf = KFold(n_splits=n_fold, shuffle=True, random_state=63)
folds = list(kf.split(nodule_ids))


for model_name, model in model_names.items():
    print(model_name)


""" ==================== Iterating through models =========================="""
for model_name, model in model_names.items():
# for model_name in model_names:
    print('='*25, ' Running Model = ', model_name , '='*25)
    
    
    '''_________________Opening New Dictionary for Saving Decision______________'''
    LBP_Image_score, LBP_Image_score_with_dia = {}, {}
    keys = ['y_true_fold','y_pred_fold', 'y_proba_fold', 'y_true_all','y_pred_all', 'y_proba_all']
    for key in keys:
        LBP_Image_score[key] = []
        LBP_Image_score_with_dia[key] = []
    
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        # if fold>=2:
        #     continue
    
        print(f"------------------ Fold {fold + 1} ----------------------")

        train_nodules = [nodule_ids[i] for i in train_idx]  # list, len 808, ['1', '10', ...]
        test_nodules = [nodule_ids[i] for i in test_idx]
        
        train_images = [img for nid in train_nodules for img in nodule_to_images[nid]]   # list of imagenames, len = 808*9 = 7272
        test_images = [img for nid in test_nodules for img in nodule_to_images[nid]]
        
    
        # Getting diameter information of test nodules_____________________________
        diameters_test_images = [image_name.split('_')[5] for image_name in test_images]
        diameters_test_nodules = diameters_test_images[0::9] # every nineth element
        diameters_test_nodules = [int(dia) for dia in diameters_test_nodules]
        
        
        
        X_train, y_train = load_selected_HOG_features_and_labels(train_images, HOG_features)
        X_test, y_test = load_selected_HOG_features_and_labels(test_images, HOG_features)

        # Convert labels to categorical one-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=2)
        y_test_cat = to_categorical(y_test, num_classes=2)

        
        '''____________________Model Making and Training____________________'''

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
 
    
        """======================= NODULE BASED PREDICTION ====================="""    
        '''_______________Getting nodule based y_test, y_test_new _________________'''
        # from 1215 (# test images) to size 135 (# test nodules)
        y_test_reshaped = y_test.reshape(-1, 9)
        
        # Check if all elements in each row are the same
        rows_are_same = np.all(y_test_reshaped == y_test_reshaped[:, 0:1], axis=1) 
        if np.all(rows_are_same):
            pass # Do nothing
            # print("All elements in every row are the same.")
        else:
            print("Rabiul Note:____All elements in every row are NOT the same. CHECK IT WHY.")
        y_test_new = np.max(y_test_reshaped, axis=1)
        
        
        
        # y_pred_prob = y_pred_prob_original
        '''_______________Getting nodule based y_pred_proba_new _________________''' 
        if fusion_style == 'average':
            y_pred_prob_reshaped = y_pred_prob.reshape(-1, 9)
            y_pred_prob_avg = y_pred_prob_reshaped.mean(axis=1)  # AVERAGING here
            y_pred_prob_new = y_pred_prob_avg
            
        elif fusion_style == 'majority':
            y_pred_prob_binary = (y_pred_prob >= 0.5).astype(int)
            y_pred_prob_reshaped = y_pred_prob_binary.reshape(-1, 9)
            y_pred_prob_majority = y_pred_prob_reshaped.sum(axis=1)/9  # MAJORITY VOTING here. 
            y_pred_prob_new = y_pred_prob_majority
           
        
        '''_______________Getting nodule based y_pred_new ______________________'''
        y_pred_new = (y_pred_prob_new >= 0.5).astype(int)
        

        # For not changing the belowing code____
        y_test = y_test_new
        y_pred_prob = y_pred_prob_new
        y_pred = y_pred_new
        
        '''____________________ ADJUST WITH DIAMETER_________________________'''
        # print(y_pred_prob)
        y_pred_prob = np.round(y_pred_prob, 2) # making 2 decimal point
        model_probs = list(y_pred_prob)
        y_pred_prob_adjusted = adjust_probability_list(diameters_test_nodules, model_probs, kde_benign, kde_malignant, alpha=0.75)
        y_pred_prob_adjusted = np.array(y_pred_prob_adjusted)
        y_pred_adjusted = ( y_pred_prob_adjusted >= 0.5).astype(int)

   
        '''_______________________ Saving Decision for Fusion__________'''
        LBP_Image_score['y_true_fold'].append(list(y_test))
        LBP_Image_score['y_pred_fold'].append(list(y_pred))
        LBP_Image_score['y_proba_fold'].append(list(y_pred_prob))

        LBP_Image_score['y_true_all'].extend(list(y_test))
        LBP_Image_score['y_pred_all'].extend(list(y_pred))
        LBP_Image_score['y_proba_all'].extend(list(y_pred_prob))
        
        
        '''_______________________ Saving Decision for Fusion with Dia Adjustment__________'''
        LBP_Image_score_with_dia['y_true_fold'].append(list(y_test))
        LBP_Image_score_with_dia['y_pred_fold'].append(list(y_pred_adjusted))
        LBP_Image_score_with_dia['y_proba_fold'].append(list(y_pred_prob_adjusted))

        LBP_Image_score_with_dia['y_true_all'].extend(list(y_test))
        LBP_Image_score_with_dia['y_pred_all'].extend(list(y_pred_adjusted))
        LBP_Image_score_with_dia['y_proba_all'].extend(list(y_pred_prob_adjusted))
        
               
        
        '''______________________Evaluating Performance_____________________'''
        performance_metrices2 = evaluate_and_print_performance(fold, y_test, y_pred, y_pred_prob, accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2)
        accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2 = performance_metrices2  
         
        '''______________ After Diameter Adjustment ________________________'''
        performance_metrices = evaluate_and_print_performance(fold, y_test, y_pred_adjusted, y_pred_prob_adjusted, accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores)
        accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities, scores = performance_metrices  
            
        
        # Plot confusion matrix_________________________________________
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, 
                    annot_kws={'size': 14, 'color': 'black', 'weight': 'bold'})
        plt.title(f'Confusion Matrix - Model_{model_name} Fold {fold + 1}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(f'{output_path}/{code_no} Model_{model_name} confusion matrix fold {fold + 1}.png')
        plt.show()
        
        # ROC Curve_______________________________________________________
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6), dpi=500)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Model_{model_name} - Fold {fold + 1}')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_path}/{code_no} Model_{model_name} ROC curve fold {fold + 1}.png')
        plt.show()
      

    '''________________Mean Scores of All Folds_____________________________'''
    results, scores = get_mean_performance_of_all_folds(performance_metrices, model_name = f'{model_name}_dia')
    save_scores_in_txt_file(output_path, code_no, scores, alpha, model_name, dia = 'Yes')
       
                
    '''________________Mean Scores of All Folds_____________________________'''
    results2, scores2 = get_mean_performance_of_all_folds(performance_metrices2, model_name = model_name)
    save_scores_in_txt_file(output_path, code_no, scores2, alpha, model_name, dia = 'No')

    """=============================Saving Results=============================="""
    # Convert results to a DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(f'{output_path}/{code_no} {model_name}_model_comparison.xlsx', index=False)
    print(f'{output_path}/{code_no} Model comparison saved to model_comparison.xlsx')
    
    
    """=================== Saving Decision ================================="""
    import pickle

    files_to_be_saved = [LBP_Image_score, LBP_Image_score_with_dia]
    file_names = ['HOG_feature_score', 'HOG_feature_score_with_dia']
    tag = model_name
    for variable, filename in zip(files_to_be_saved, file_names):
        # saving pickle file_____________________
        with open(f'{output_path}/{filename}_{tag}.pkl', 'wb') as file:
            pickle.dump(variable, file)
