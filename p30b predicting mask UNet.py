import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

# Load the trained U-Net model
model_path = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\1. Restart Coding\Code Outputs\rc18 UNet Seg 64\unet_model_fold_5.h5"
model = tf.keras.models.load_model(model_path)

# Paths
source_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules"  # Replace with actual path
destination_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p30 SPIE data saving\2D Nodules Predicted Mask" # Replace with actual path
os.makedirs(destination_folder, exist_ok=True)  # Create destination folder if not exists

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """Load an image, resize, and normalize it for U-Net."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading: {image_path}")
        return None
    
    image = cv2.resize(image, target_size)  # Resize to model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image

def predict_and_save_mask(image_path, output_path):
    """Predict mask using U-Net model and save the result."""
    image = load_and_preprocess_image(image_path)
    if image is None:
        return
    
    # Predict mask
    predicted_mask = model.predict(image)[0]  # Get first output
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Threshold & convert to binary
    
    # Save mask
    cv2.imwrite(output_path, predicted_mask)

# Process all images
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(source_folder, image_file)
    output_path = os.path.join(destination_folder, f"mask_{image_file}")
    predict_and_save_mask(image_path, output_path)

print(f"✅ All masks saved to: {destination_folder}")
