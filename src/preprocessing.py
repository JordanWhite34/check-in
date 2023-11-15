# preprocessing.py
import os
import cv2
import numpy as np
import torch
from config import PREPROCESSING_CONFIG, DATA_PATHS


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image


def transform_image(image):
    # Resize image using OpenCV
    resized_image = cv2.resize(image, PREPROCESSING_CONFIG['image_size'])

    # Convert to float and normalize
    resized_image = resized_image.astype(np.float32) / 255.
    mean = np.array(PREPROCESSING_CONFIG['mean'])
    std = np.array(PREPROCESSING_CONFIG['std'])
    normalized_image = (resized_image - mean) / std

    return normalized_image


def preprocess_directory(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory, filename)

            try:
                image = load_image(input_path)
                transformed_image = transform_image(image)

                # Convert normalized image back to 0-255 range for saving
                transformed_image = np.clip(transformed_image, 0, 1)
                transformed_image = (transformed_image * 255).astype(np.uint8)

                # Save the image using OpenCV
                cv2.imwrite(output_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    for category in ['clean', 'messy']:
        input_dir = os.path.join(DATA_PATHS['raw'], category)
        output_dir = os.path.join(DATA_PATHS['processed'], category)

        preprocess_directory(input_dir, output_dir)
