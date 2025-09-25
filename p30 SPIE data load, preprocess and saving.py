""" p30 SPIE data load, preprocess and saving
Task: Loading, Preprocessing and Saving Data
Dataset: SPIE-AAPM
Date: Feb 23, 2025 By: Md Rabiul Islam, ECEN, TAMU
"""

"""========================== Part 1: Importing Libraries =================="""
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
from skimage.measure import label, regionprops


# Parameters__________________
new_spacing=(1, 1, 1)
box_size = 32
half_box_size = box_size // 2
new_spacing=(1, 1, 1)
croping_shape = (32, 32, 32)

# Paths________________________
code_output = 'Code Outputs/p30 SPIE data saving'
output_2d_images = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules"
output_2d_masks_predicted = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules Predicted Mask"
output_2d_9slices_img = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules Slices"
output_3d_images = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\3D Nodules Volume"


"""======================= Part 2: User Defined Functions =================="""
def get_dicom_images(data_path, root_folder):
    '''
    Input: root_folder = a folder that contains nested folder of order 3 and finally get dicom files (.dcm)
    That means, root_folder --> another_folder --> another_folder --> .dcm files
    
    Output: Images (as numpy array)
    '''
    dicom_files = []
    root_folder_path = os.path.join(data_path, root_folder)
    for root, dirs, files in os.walk(root_folder_path):
        # print(files)
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
                
    all_slices = [pydicom.dcmread(f'{path}') for path in dicom_files]
    all_slices.sort(key = lambda x: int(x.InstanceNumber))
    # all_slices_images = [dataset.pixel_array for dataset in all_slices]
    
    return all_slices #_images


def correct_HU(image_data, level, window):
    '''Input: Data file, Output: numpy array'''
    
    intercept = int(image_data[40].RescaleIntercept) # normally -1024, but may change
    
    # converting Data to Arrary
    image_volume = np.array([s.pixel_array for s in image_data])
    
    # image_volume[image_volume <= -1000] = 0, Not needed here
    image_volume = image_volume + intercept # for simplicity we can use, image_volume = image_volume - 1024.0

    min_value = level - window // 2
    max_value = level + window // 2
    image_volume = np.clip(image_volume, min_value, max_value)
    return image_volume # returning np array


def normalize_volume_0_1(volume):
    max_value, min_value = np.max(volume), np.min(volume)
    normalized_volume = (volume - min_value) / (max_value - min_value)
    normalized_volume = np.clip(normalized_volume, 0, 1)
    return normalized_volume


def get_centroid(a_dicom_foldername, nodule_id):
    nodule_annotation = annotation_df.loc[annotation_df['Scan Number']==a_dicom_foldername]
    nodule_coordinates = nodule_annotation.iloc[nodule_id:nodule_id+1, 2:4] # as 2nd and 3rd colum contains the coordinates
    x, y = nodule_coordinates.iloc[:, 0].values[0].split(',') # x, y string
    z = nodule_coordinates.iloc[:, 1].values[0] # z as float
    x, y, z = map(int, [x, y, z]) # making int
    return x, y, z


def resample_and_crop_volume_d2(nodule_volume, real_centroid, original_spacing, new_spacing, croping_shape):
    """
    Resample a 3D volume from original spacing (e.g. 0.703125, 0.703125, 2.5) to new spacing (e.g. 1, 1, 1)
    crop a cube (40*40*40 mm^3) around the centroid which means croping_shape (40*40*40). 
    
    Parameters:
        nodule_volume (numpy array): The original 3D Nodule volume in (x, y, z) order.
        real_centroid (tuple): The original centroid coordinates in (x, y, z) order.
        mask_volume (numpy array): The original 3D Mask volume in (x, y, z) order.
        original_spacing (tuple): The original spacing of the volume in (x, y, z) order.
        new_spacing (tuple): The desired voxel spacing of the volume in (x, y, z) order.
        croping_shape (tuple): The desired final 3D shape of Nodule and Mask volume
        
    
    Returns:
        numpy array: The resampled and cropped 3D volume (cube_size x cube_size x cube_size).
    """
    
    '''________________________ Resampling _________________________________'''
    # Compute the resampling factors for each dimension (x, y, z)
    resample_factors = np.array(original_spacing) / np.array(new_spacing)

    # Resample the volume
    resampled_volume = zoom(nodule_volume, resample_factors, order=3)  # 'order=3' for spline interpolation
    # resampled_mask = zoom(mask_volume, resample_factors, order=0)
    
    '''________________________ Croping ____________________________________'''        
    # Adjust the centroid to the new resampled volume
    resampled_centroid = np.array(real_centroid) * resample_factors
    x_center, y_center, z_center = map(int, np.round(resampled_centroid))
    
    # Define the crop boundaries (centered around the new centroid)
    crop_x, crop_y, crop_z = croping_shape
    half_x = int(crop_x / 2)
    half_y = int(crop_y / 2)
    half_z = int(crop_z / 2)
        
    # Define the boundaries of the cube to extract (ensure they are within volume bounds)
    x_min = max(x_center - half_x, 0)
    x_max = min(x_center + half_x, resampled_volume.shape[0]) # + 1 because desired shape is 71, a odd number
    y_min = max(y_center - half_y, 0)
    y_max = min(y_center + half_y, resampled_volume.shape[1])
    z_min = max(z_center - half_z, 0)
    z_max = min(z_center + half_z, resampled_volume.shape[2])
    
    # conversion needed for second dataset.__________IMPORTANT
    cropped_nodule = resampled_volume[y_min:y_max, x_min:x_max, z_min:z_max]

    return cropped_nodule


def get_nine_slices(cube): 
    cube_size = cube.shape[0]
    mid_x = mid_y = mid_z = cube_size // 2
    
    # Middle 3 slices 
    coronal = cube[:, mid_y, :]     # front-view
    sagittal = cube[mid_x, :, :]    # left-view
    axial = cube[:, :, mid_z]       # top-view
    
    # Diagonal 6 slices 
    plane_1 = np.array([cube[i, i, :] for i in range(cube_size)])
    plane_2 = np.array([cube[i, cube_size-1-i, :] for i in range(cube_size)])

    plane_3 = np.transpose(np.array([cube[i, :, i] for i in range(cube_size)]))
    plane_4 = np.transpose(np.array([cube[i, :, cube_size-1-i] for i in range(cube_size)]))

    plane_5 = np.array([cube[:, i, i] for i in range(cube_size)])
    plane_6 = np.array([cube[:, i, cube_size-1-i] for i in range(cube_size)])
    
    nine_slices = np.stack((coronal, sagittal, axial, plane_1, plane_2, plane_3, plane_4, plane_5, plane_6), axis=0)
    return nine_slices


def calculate_diameter(mask):
    # Label connected components (regions)
    labeled_mask = label(mask)
    # Get properties of the largest region (nodule)
    regions = regionprops(labeled_mask)
    if regions:
        largest_region = max(regions, key=lambda x: x.area)
        # Get the coordinates of the bounding box of the nodule
        minr, minc, maxr, maxc = largest_region.bbox
        # Calculate the diameter (distance between furthest points along the region)
        diameter_pixels = np.linalg.norm([maxr - minr, maxc - minc])
        # Convert to mm (scale factor = 0.5 mm/pixel)
        diameter_mm = diameter_pixels * 0.5
        return diameter_mm
    return 0.0  # Return 0 if no nodule found

''' ================== Part 3: Loading Dicom Images ========================'''
data_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Dataset\SPIE-AAPM dataset\SPIE-AAPM Lung CT Challenge"

# Make a list of folder like [CT-Training-BE001, CT-Training-BE02, ...]
patient_list = os.listdir(data_path)

# Printing name of folder 
for idx, name in enumerate(patient_list):
    print(f'{idx+1}. {name}')

# Loading all 3D volume Images ________________________________________________
all_3d_volume = []
print(f'Loading CT Scans___________________Rabiul_Islam')
for root_folder in tqdm(patient_list):
    a_3d_volume = get_dicom_images(data_path, root_folder)
    all_3d_volume.append(a_3d_volume)
print(f'{len(all_3d_volume)} DICOM files are loaded.')

    
'''________________________Loading Label File_______________________________'''
# Loading Label, benign malignant
label_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Code 8 SPIE dataset\Dataset\2. Rabiul Preprocessed APIE-AAPM\Rabiul_Label.xlsx"
annotation_df = pd.read_excel(label_path) 


# Load the trained U-Net model trained with dataset 1
model_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc18 UNet Seg 64\unet_model_fold_5.h5"
model = tf.keras.models.load_model(model_path)


"""========================== Part 4: Preprocessing ========================"""
nodule_count = 0
for idx, a_3d_volume in enumerate(all_3d_volume):
    a_dicom_foldername = patient_list[idx]
    
    # Preprocess 1___________________ HU correction___________________
    vol_HU_corrected = correct_HU(a_3d_volume, level=-200, window=1200)
    vol_HU_corrected = np.transpose(vol_HU_corrected, (1, 2, 0)) #reshaping from (z, x, y) to (x, y, z)
    # print(vol_HU_corrected.shape)
    
    n_nodules = annotation_df['Scan Number'].str.contains(a_dicom_foldername).sum()
    
    #________________________iterating by nodule_________________________
    # 13 patients have 2 nodules, so number of total nodules = 70+13 = 83
    for nodule_id in range(n_nodules):
        

        # Preprocess 2 ________________geting centroid_________________________
        x, y, z = get_centroid(a_dicom_foldername, nodule_id)
        # print(x, y, z)
        
        # ploting rectangle to see nodule
        
        # plt.imshow(vol_HU_corrected[:, :, z-1 ])
        # plt.scatter(x, y, color='red', s=1)
        # rect = plt.Rectangle((x - half_box_size, y - half_box_size), box_size, box_size,
        #                      edgecolor='red', facecolor='none', linewidth=1)
        # plt.gca().add_patch(rect)
        # plt.axis('off')
        # plt.show()

        # Preprocess 3__________________ Resampling and croping _______________
        nodule_volume = normalize_volume_0_1(vol_HU_corrected)
        real_centroid = (x, y, z)
        a_slice = a_3d_volume[0] 
        spacing_x, spacing_y, spacing_z = float(a_slice.PixelSpacing[0]), float(a_slice.PixelSpacing[1]), float(a_slice.SliceThickness)
        original_spacing = (spacing_x, spacing_y, spacing_z)
        

        nodule_res_crop = resample_and_crop_volume_d2(nodule_volume, real_centroid, original_spacing, new_spacing, croping_shape)

        # showing resampled and croped nodule
        # plt.imshow(nodule_res_crop[:, :, 15])
        # plt.show()
        
        # Preprocess 4 _________________ Extracting 9 Views ___________________
        nodule_nine_slices = get_nine_slices(nodule_res_crop)
        # for i in range(9):
        #     plt.imshow(nodule_nine_slices[i])
        #     plt.show()
        
        
        # Preprocess 5_________________ Predicting Mask with UNet________________________
        nodule_nine_slices_64 = np.array([resize(img, (64, 64), anti_aliasing=True) for img in nodule_nine_slices])
        nodule_nine_slices_tmp = np.expand_dims(nodule_nine_slices_64, axis=-1)  # Shape: (9, 32, 32, 1)
        predicted_mask_9v = model.predict(nodule_nine_slices_tmp)
        predicted_mask_9v = (predicted_mask_9v > 0.5).astype(np.uint8)
        
        # Preprocess 6___________Estimating diameter and skip nodules <3mm_____
        diameters = []
        for i in range(predicted_mask_9v.shape[0]):
            # Extract the 2D slice from the 3D array
            slice_mask = predicted_mask_9v[i, :, :, 0]  # Shape: (32, 32)
            diameter = calculate_diameter(slice_mask)
            diameters.append(diameter)
        dia = int(sum(diameters)/len(diameters))
        
        if dia>= 3: # if dia> 3 mm, only then save and count nodule
            nodule_count += 1
            # saving nodules_______________________________________________________
            print(f"Saving {a_dicom_foldername} nodule's 2D_Images, 9V_Slices and 3D_Volume")
            label_df = annotation_df[annotation_df.iloc[:, 0] == a_dicom_foldername]
            label_type = label_df.iloc[nodule_id, -1]
            cancer_label = 'Y' if 'malignant' in label_type else 'N'
            
            # saving 9 Slices as Image ______________________________
            for i in range(9):
                # Saving 2D Images _______________
                nodule_name = "nid_{}_s_{}_dia_{}_p_{}_n_{}_c_{}.png".format(nodule_count, i+1, dia, a_dicom_foldername, nodule_id+1, cancer_label)
                img =  nodule_nine_slices[i, :, :]
                img = resize(img, (64, 64), anti_aliasing=True)
                nodule_path = os.path.join(output_2d_images, nodule_name)
                plt.imsave(nodule_path, img, cmap='gray')
                
                # Saving Predicted Masks__________
                nodule_path = os.path.join(output_2d_masks_predicted, nodule_name)
                mask = predicted_mask_9v[i, :, :, 0]
                # mask = resize(mask, (64, 64), anti_aliasing=True)
                plt.imsave(nodule_path, mask, cmap='gray')
            
            # Saving 9 Slices as Array ______________________________
            nodule_name = "n2D_{}_dia_{}_p_{}_n_{}_c_{}.npy".format(nodule_count, dia, a_dicom_foldername, nodule_id+1, cancer_label)
            nodule_path = os.path.join(output_2d_9slices_img, nodule_name)
            np.save(nodule_path, nodule_nine_slices)
            
            # Saving 3D nodule volume 32, 32, 32
            nodule_name = "n3D_{}_dia_{}_p_{}_n_{}_c_{}.npy".format(nodule_count, dia, a_dicom_foldername, nodule_id+1, cancer_label)
            nodule_path = os.path.join(output_3d_images, nodule_name)
            np.save(nodule_path, nodule_res_crop)
      
        else:
            print(f"dia of a nodule in {a_dicom_foldername} is <= 3 mm. Ignored")