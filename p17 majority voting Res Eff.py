'''
=============================== ABOUT CODING ==================================
Date: Feb 18, 2025. Copy of p2 9v searching for best model.py________________________________________
Task:  MAJORITY VOTING ResNet, EfficientNet
fusion_style = 'majority', 'average'
________________________________________________________

'''

code_no = 'p17' # just checking a simple 2D CNN based model accuracy
output_path = 'Code Outputs/p17 majority voting Res Eff'
malignancy_3 ='ignored' #'benign' 'malignant'
fusion_style = 'majority'

"""============================= Importing Libraries ====================== """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNet, NASNetMobile, EfficientNetV2B0, InceptionV3, ConvNeXtTiny
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
from skimage import io, img_as_float, measure, color
import csv
import tensorflow as tf
import tensorflow.keras as K
from scipy.stats import gaussian_kde
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from lung_utility import create_model, adjust_probability_list, evaluate_and_print_performance, get_mean_performance_of_all_folds, save_scores_in_txt_file



# Training parameters
n_fold = 7
n_batch = 16
n_epoch = 50
n_patience = 10

# model_name = "EfficientNetV2B0"
model_names = ['EfficientNetV2B0', 'ResNet50'] # ,'ResNet50', 'VGG16', 'VGG19' 'ResNet50', 'EfficientNetV2B0', 'InceptionV3', 'MobileNet', 'NASNetMobile', 'DenseNet121', 'ConvNeXtTiny'
n_freeze = 50 



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

'''_____________Loading image filenames grouped by NODULEs____________________'''
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



def load_images_and_labels(image_list, folder_path):
    X, y = [], []
    for img_file in image_list:
        img_path = os.path.join(folder_path, img_file)
        img = load_img(img_path)
        img_array = np.uint8(img_to_array(img))
        X.append(img_array)

        # Extract label
        label = 1 if "_c_Y" in img_file else 0
        y.append(label)
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



"""============================== Pretrained Models ========================"""





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


# diameters_all = []
""" ==================== Iterating through models ============================"""
for model_name in model_names:
    print('='*25, ' Running Model = ', model_name , '='*25)
    
    
    '''_________________Opening New Dictionary for Saving Decision______________'''
    DEEP_Image_score, DEEP_Image_score_with_dia = {}, {}
    keys = ['y_true_fold','y_pred_fold', 'y_proba_fold', 'y_true_all','y_pred_all', 'y_proba_all']
    for key in keys:
        DEEP_Image_score[key] = []
        DEEP_Image_score_with_dia[key] = []
        
    # 
    for fold, (train_idx, test_idx) in enumerate(folds):
        # if fold>=1:
        #     continue

        train_nodules = [nodule_ids[i] for i in train_idx]  # list, len 808, ['1', '10', ...]
        test_nodules = [nodule_ids[i] for i in test_idx]
        
        train_images = [img for nid in train_nodules for img in nodule_to_images[nid]]   # list of imagenames, len = 808*9 = 7272
        test_images = [img for nid in test_nodules for img in nodule_to_images[nid]]
        
        # Getting diameter information of test nodules_____________________________
        diameters_test_images = [image_name.split('_')[5] for image_name in test_images]
        diameters_test_nodules = diameters_test_images[0::9] # every nineth element
        diameters_test_nodules = [int(dia) for dia in diameters_test_nodules]
        # diameters_all.extend(diameters_test_nodules) 
        
#     break   
# np.save('diameters_all.npy', np.array(diameters_all))
# #%%

        X_train, y_train = load_images_and_labels(train_images, image_folder)
        X_test, y_test = load_images_and_labels(test_images, image_folder)
        # plt.imshow(X_test[9])
        
        '''___________________Processing Before Model_______________________'''
        print(f"------------------ Fold {fold + 1} ----------------------")
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # dia_train, dia_test = diameters[train_index], diameters[test_index]

        # Convert labels to categorical one-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=2)
        y_test_cat = to_categorical(y_test, num_classes=2)

        # Data augmentation
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        datagen.fit(X_train)        
        
        
        '''____________________Model Making and Training____________________'''
        # Initialize model
        model = create_model(model_name, n_freeze)
        # print(model.summary())
        
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=n_patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        model_checkpoint = ModelCheckpoint(f'{output_path}/{code_no} Model_{model_name}_best_model_fold_{fold + 1}.h5', monitor='val_accuracy', save_best_only=True, mode='max')

        # Train the model
        history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=n_batch), 
                            epochs=n_epoch, 
                            validation_data=(X_test, y_test_cat),
                            callbacks=[early_stopping, reduce_lr, model_checkpoint])

        
        '''______________________Saving Training Scores_____________________'''
        # Save history to CSV
        csv_filename = f'{output_path}/{code_no} Model_{model_name}_fold_{fold + 1}_history.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for epoch in range(len(history.history['accuracy'])):
                writer.writerow({
                    'epoch': epoch + 1,
                    'accuracy': history.history['accuracy'][epoch],
                    'loss': history.history['loss'][epoch],
                    'val_accuracy': history.history['val_accuracy'][epoch],
                    'val_loss': history.history['val_loss'][epoch]
                })

        
        '''______________________Testing With Best Model________________________'''
        # Load best model
        best_model = load_model(f'{output_path}/{code_no} Model_{model_name}_best_model_fold_{fold + 1}.h5')
        # Evaluate the model on test data
        y_pred_prob = best_model.predict(X_test)
        y_pred_prob_original = y_pred_prob
        y_pred_prob = y_pred_prob[:, 1] # shape n_row, 1. considering only probability of being malignant
        

        
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
        DEEP_Image_score['y_true_fold'].append(list(y_test))
        DEEP_Image_score['y_pred_fold'].append(list(y_pred))
        DEEP_Image_score['y_proba_fold'].append(list(y_pred_prob))

        DEEP_Image_score['y_true_all'].extend(list(y_test))
        DEEP_Image_score['y_pred_all'].extend(list(y_pred))
        DEEP_Image_score['y_proba_all'].extend(list(y_pred_prob))
        
        
        '''_______________________ Saving Decision for Fusion with Dia Adjustment__________'''
        DEEP_Image_score_with_dia['y_true_fold'].append(list(y_test))
        DEEP_Image_score_with_dia['y_pred_fold'].append(list(y_pred_adjusted))
        DEEP_Image_score_with_dia['y_proba_fold'].append(list(y_pred_prob_adjusted))

        DEEP_Image_score_with_dia['y_true_all'].extend(list(y_test))
        DEEP_Image_score_with_dia['y_pred_all'].extend(list(y_pred_adjusted))
        DEEP_Image_score_with_dia['y_proba_all'].extend(list(y_pred_prob_adjusted))
        
        
        '''______________________Evaluating Performance_____________________'''
        performance_metrices2 = evaluate_and_print_performance(fold, y_test, y_pred, y_pred_prob, accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2)
        accuracies2, precisions2, recalls2, f1_scores2, roc_aucs2,  sensitivities2, specificities2, scores2 = performance_metrices2  
         
        '''______________ After Diameter Adjustment ________________________'''
        performance_metrices = evaluate_and_print_performance(fold, y_test, y_pred_adjusted, y_pred_prob_adjusted, accuracies, precisions, recalls, f1_scores, roc_aucs,  sensitivities, specificities, scores)
        accuracies, precisions, recalls, f1_scores, roc_aucs, sensitivities, specificities, scores = performance_metrices  
            
        '''________________________ Plotting Curves ________________________'''
        
        # Plot training & validation ACCURACY values_______________________
        plt.figure(figsize=(12, 4), dpi=500)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.title(f'Model Accuracy - Model_{model_name} Fold {fold + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training & validation LOSS values__________________________
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title(f'Model Loss - Model_{model_name} Fold {fold + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/{code_no} Model_{model_name} accuracy and loss curve fold {fold + 1}.png')
        plt.show()
        
        
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

    files_to_be_saved = [DEEP_Image_score, DEEP_Image_score_with_dia]
    file_names = ['DEEP_Image_score', 'DEEP_Image_score_with_dia']
    tag = model_name[:3]
    for variable, filename in zip(files_to_be_saved, file_names):
        # saving pickle file_____________________
        with open(f'{output_path}/{filename}_{tag}.pkl', 'wb') as file:
            pickle.dump(variable, file)

