import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


# Loading a DICOM volme________________________________________________________
dataset_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Code 8 SPIE dataset\Dataset\1. SPIE-AAPM Main Dataset"
dicom_folderpath = os.path.join(dataset_path, 'Lung CT Scans DICOM')
a_dicom_foldername = "LUNGx-CT016"


def get_dicom_images(dataset_path, a_dicom_foldername):
    ''' Input:  dataset_path = a folder that contains nested folder of order 3 and finally get dicom files (.dcm)
                a_dicom_foldername --> another_folder --> another_folder --> .dcm files
        Output: Images (as numpy array)'''
    dicom_files = []
    root_folder_path = os.path.join(dataset_path, a_dicom_foldername)
    for root, dirs, files in os.walk(root_folder_path):
        # print(files)
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))            
    all_slices = [pydicom.dcmread(f'{path}') for path in dicom_files]
    all_slices.sort(key = lambda x: int(x.InstanceNumber))
    # all_slices_images = [dataset.pixel_array for dataset in all_slices]
    return all_slices #_images

dicom_data = get_dicom_images(dicom_folderpath, a_dicom_foldername) 
dicom_img = np.array([s.pixel_array for s in dicom_data])


# Changing HU__________________________________________________________________
def correct_HU(dicom_data, level, window):
    '''Input: imageData file, Output: numpy array'''
    
    intercept = int(dicom_data[40].RescaleIntercept) # normally -1024, but may change
    
    # converting Data to Arrary
    image_volume = np.array([s.pixel_array for s in dicom_data])
    
    # image_volume[image_volume <= -1000] = 0 #Not needed here
    image_volume = image_volume + intercept # for simplicity we can use, image_volume = image_volume - 1024.0

    min_value = level - window // 2
    max_value = level + window // 2
    image_volume = np.clip(image_volume, min_value, max_value)
    return image_volume # returning np array


LEVEL, WINDOW = -500, 1500
dicom_img_hu = correct_HU(dicom_data, level=LEVEL, window=WINDOW)


# Plotting_____________________________________________________________________
# looking some normal slices
for i in range(50):
    plt.imshow(dicom_img[i])
    plt.show() 

# looking some HU corrected slices
for i in range(50):
    plt.imshow(dicom_img_hu[i])
    plt.show()

# normal slice vs HU corrected slice
SLICE_N = 70
plt.subplot(1,2,1)
plt.imshow(dicom_img[SLICE_N])
plt.title('Normal CT Scan')
plt.subplot(1,2,2)
plt.imshow(dicom_img_hu[SLICE_N])
plt.title('HU corrected CT Scan')
plt.show()

# Ploting Lung with Nodule & Centroid _________________________________________
annotation_filename = "TestSet_NoduleData_PublicRelease_wTruth.xlsx"
annotation = pd.read_excel(os.path.join(dataset_path, annotation_filename)) 
nodule_annotation = annotation.loc[annotation['Scan Number']==a_dicom_foldername]
nodule_coordinates = nodule_annotation.iloc[:, 2:4] # as 2nd and 3rd colum contains the coordinates
x, y = nodule_coordinates.iloc[:, 0].values[0].split(',') # x, y string
z = nodule_coordinates.iloc[:, 1].values[0] # z as float
x, y, z = map(int, [x, y, z]) # making int


# changing axis of dicom_img_hu from z,x,y to x,y,z
dicom_img_hu_r = np.transpose(dicom_img_hu, (1, 2, 0))
box_size = 32
half_box_size = box_size // 2

plt.imshow(dicom_img_hu_r[:, :, z-1]) # as arrary index start from 0
plt.scatter(x-1, y-1, color='red', s=2)
rect = plt.Rectangle((x-1 - half_box_size, y-1 - half_box_size), box_size, box_size, edgecolor='red', facecolor='none', linewidth=1)
plt.gca().add_patch(rect)
plt.axis('off')
plt.show()


# cropping nodule______________________________________________________________
def normalize_volume_0_1(volume):
    max_value, min_value = np.max(volume), np.min(volume)
    normalized_volume = (volume - min_value) / (max_value - min_value)
    normalized_volume = np.clip(normalized_volume, 0, 1)
    return normalized_volume

dicom_img_hu_r = normalize_volume_0_1(dicom_img_hu_r)

# conversion from coordinate measurement x,y to image measurement row, col__IMPORTANT
row_low = y-16 # I want to crop 32, 32
row_high= y+16
col_low = x-16
col_high = x+16
plt.imshow(dicom_img_hu_r[row_low:row_high, col_low:col_high, z-1])
plt.show()


# Resampling, croping then showing Nodule _____________________________________
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


nodule_volume = dicom_img_hu_r
real_centroid = (x, y, z)
a_slice = dicom_data[0] 
spacing_x, spacing_y, spacing_z = float(a_slice.PixelSpacing[0]), float(a_slice.PixelSpacing[1]), float(a_slice.SliceThickness)
original_spacing = (spacing_x, spacing_y, spacing_z)
new_spacing=(1, 1, 1)
croping_shape = (32, 32, 32)

nodule_res_crop = resample_and_crop_volume_d2(nodule_volume, real_centroid, original_spacing, new_spacing, croping_shape)

plt.imshow(nodule_res_crop[:, :, 15])
plt.show()
