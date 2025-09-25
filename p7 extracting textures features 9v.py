'''
Copy of extracting all features. Date February 8, 2025.

Extracting and Saving all the FEATURES from Input Images & Masks
Date: July 21, 2024
FEATURES ARE:
    1. LBP Images       <----- Input Images + Masks
    2. GLCM Features    <----- Input Images
    3. HOG Images       <----- Input Images
    4. HOG Features     <----- Input Images
'''

"""=====================Part 1: Importing Libraries ========================"""
import os
import numpy as np
import pandas as pd
from skimage import io, img_as_float, measure, color
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import csv
import tensorflow as tf
import tensorflow.keras as K

from tqdm import tqdm


"""======================= Loading Data  ==================================="""

def get_dataset_based_label(malignancy_level):
    '''
    dataset = 1 meaning, Benign = 1, 2      Malignant = 4, 5
    dataset = 2 meaning, Benign = 1, 2, 3   Malignant = 4, 5
    dataset = 3 meaning, Benign = 1, 2      Malignant = 3, 4, 5
    '''
    if malignancy_level in [1, 2]:            
        dataset_1_label = 0
        dataset_2_label = 0
        dataset_3_label = 0
    elif malignancy_level in [4, 5]:
        dataset_1_label = 1
        dataset_2_label = 1
        dataset_3_label = 1
    elif malignancy_level in [3]:
        dataset_1_label = 2  # Unknown = 2
        dataset_2_label = 0  # Benign = 0
        dataset_3_label = 1  # Malignant = 1
    return dataset_1_label, dataset_2_label, dataset_3_label


def get_LBP_images(image_gray):
    lbp_image = local_binary_pattern(image_gray, P=24, R=3, method="uniform")
    lbp_image_amplified = (lbp_image * 255/np.max(lbp_image.ravel())).astype(np.uint8)
    return lbp_image_amplified


def extract_GLCM_features(image_gray, distances, angles):
    image_255 = (image_gray * 255).astype(np.uint8)
    glcm = graycomatrix(image_255, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    a = list(graycoprops(glcm, 'contrast').flatten())
    b = list(graycoprops(glcm, 'dissimilarity').flatten())
    c = list(graycoprops(glcm, 'homogeneity').flatten())
    d = list(graycoprops(glcm, 'energy').flatten())
    e = list(graycoprops(glcm, 'correlation').flatten()) 
    features.append(a+b+c+d+e)
    features = features
    features = [[round(value, 2) for value in sublist] for sublist in features]
    return features[0]


def get_HOG_images(image_gray):  
    hog_features, hog_image = hog(image_gray, 
                                          orientations=9, 
                                          pixels_per_cell=(8,8), 
                                          cells_per_block=(2,2), 
                                          block_norm='L2-Hys', 
                                          visualize=True)
    
    hog_image_int = (hog_image * 255/np.max(hog_image.ravel())).astype(np.uint8)
    return hog_image_int


def extract_hog_features(image_gray):
    hog_features = hog(image_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    hog_features_rounded = [round(item, 3) for item in hog_features]
    return hog_features_rounded


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    
def load_data_and_extract_features(input_images_folder, output_features_folder):
    
    images_gray = []
    # images_color = []
    labels = []
    LBP_images = []
    GLCM_features = []    
    HOG_images = []
    HOG_features = []    
    
    
    # Making colum name for GLCM features_________run it once, thats why outside for loop
    colum_name_GLCM = ['nodule_id']
    feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for feature in feature_names:  # 8 groups of features, numbered 0 to 7
        for no in range(1,25):
            colum_name_GLCM.append( f'{feature}_{no}')
    colum_name_GLCM.extend(['d1_label', 'd2_label', 'd3_label'])
    
    
    # making colum name for HOG features
    colum_name_HOG = ['nodule_id']
    for idx in range(1764):
        colum_name_HOG.append(f'HOG_{idx+1}')
    colum_name_HOG.extend(['d1_label', 'd2_label', 'd3_label'])
    
    
    # for filename in os.listdir(input_images_folder)[:100]:
    for i, filename in enumerate(tqdm(os.listdir(input_images_folder), desc="Processing Images")):
        '''_______________________Loading Input Images________________'''
        image_gray = io.imread(os.path.join(input_images_folder, filename), as_gray=True)
        images_gray.append(image_gray)
        
        '''_______________________Getting the LABEL___________________'''
        malignancy_level = int(filename.split('_')[11])
        # print(filename, malignancy_level)
        d1_label, d2_label, d3_label = get_dataset_based_label(malignancy_level)
        labels.append([d1_label, d2_label, d3_label])

        
        # '''_______________________Getting LBP Images___________________'''        
        # LBP_image = get_LBP_images(image_gray)
        # LBP_images.append(LBP_image)
        
        # # Save the LBP image__________
        # LBP_img_folder = os.path.join(output_features_folder, 'LBP Images')
        # make_folder(LBP_img_folder)
        # LBP_image_name = os.path.join(LBP_img_folder, filename)
        # io.imsave(LBP_image_name, LBP_image)

        
        '''_______________________Getting GLCM Features_________________'''
        GLCM_feature = extract_GLCM_features(image_gray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]) #0, np.pi/4, np.pi/2, 3*np.pi/4
        GLCM_features.append([filename] + GLCM_feature + [d1_label, d2_label, d3_label])

        
        # '''_______________________Getting HOG Images___________________'''
        # HOG_image = get_HOG_images(image_gray)
        # HOG_images.append(HOG_image)
        
        # # saving_HOG images__________________
        # HOG_img_folder = os.path.join(output_features_folder, 'HOG Images')
        # make_folder(HOG_img_folder)
        # HOG_image_name = os.path.join(HOG_img_folder, filename)
        # io.imsave(HOG_image_name, HOG_image)
        
        
        # '''_______________________Getting HOG Fetures___________________'''
        # HOG_feature = extract_hog_features(image_gray)
        # HOG_features.append([filename] + HOG_feature + [d1_label, d2_label, d3_label])

    
    # saving GLCM features_____________________
    GLCM_features_df = pd.DataFrame(GLCM_features, columns = colum_name_GLCM) 
    GLCM_features_df.to_excel(f'{output_features_folder}/GLCM_features.xlsx', index=False)
    
    # # saving HOG features______________________
    # HOG_features_df = pd.DataFrame(HOG_features, columns = colum_name_HOG) 
    # HOG_features_df.to_excel(f'{output_features_folder}/HOG_features.xlsx', index=False)

    
    # return images_gray, labels, LBP_images, GLCM_features_df, HOG_images, HOG_features_df
    return images_gray, labels, GLCM_features_df


"""========================= MAIN USER DEFINED FUNCTIOIN ======================"""
input_images_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc17 save 3D nodule 32\2D Nodules 64\Images"
output_features_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p7 extracting textures features 9v"

images_gray, labels,  GLCM_features_df = load_data_and_extract_features(input_images_folder, output_features_folder)