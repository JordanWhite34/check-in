# preprocessing.py
import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from config import PREPROCESSING_CONFIG, DATA_PATHS


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image


def transform_image(image):
    resized_image = cv2.resize(image, PREPROCESSING_CONFIG['image_size'])
    # Normalize the image
    resized_image = resized_image.astype(np.float32) / 255.0
    mean = np.array(PREPROCESSING_CONFIG['mean'], dtype=np.float32)
    std = np.array(PREPROCESSING_CONFIG['std'], dtype=np.float32)
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


def split_data(source, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15):
    files = os.listdir(source)
    train_files, test_files = train_test_split(files, test_size=1 - train_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (train_size + val_size),
                                              random_state=42)

    for file in train_files:
        shutil.copy(os.path.join(source, file), train_dir)
    for file in val_files:
        shutil.copy(os.path.join(source, file), val_dir)
    for file in test_files:
        shutil.copy(os.path.join(source, file), test_dir)


if __name__ == "__main__":
    categories = ['clean', 'messy']
    for category in categories:
        input_dir = os.path.join(DATA_PATHS['raw'], category)
        train_dir = os.path.join(DATA_PATHS['processed'], 'train', category)
        val_dir = os.path.join(DATA_PATHS['processed'], 'val', category)
        test_dir = os.path.join(DATA_PATHS['processed'], 'test', category)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        split_data(input_dir, train_dir, val_dir, test_dir)

        # Now preprocess each split
        preprocess_directory(train_dir, train_dir)
        preprocess_directory(val_dir, val_dir)
        preprocess_directory(test_dir, test_dir)

        print("preprocessing complete")