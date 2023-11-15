# preprocessing.py
import os
import cv2
import torch
from torchvision import transforms
from config import PREPROCESSING_CONFIG, DATA_PATHS


# Function to load an image using OpenCV and convert it to RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image


# Function to apply transformations to an image
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(PREPROCESSING_CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=PREPROCESSING_CONFIG['mean'],
            std=PREPROCESSING_CONFIG['std']
        )
    ])
    return transform(image)


# Function to preprocess all images in a directory
def preprocess_directory(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Add other formats as needed
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory, filename)

            try:
                image = load_image(input_path)
                transformed_image = transform_image(image)
                torch.save(transformed_image, output_path.replace('.jpg', '.pt'))  # Saving as PyTorch tensor
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Main function to run the preprocessing
if __name__ == "__main__":
    for category in ['clean', 'messy']:
        input_dir = os.path.join(DATA_PATHS['raw'], category)
        output_dir = os.path.join(DATA_PATHS['processed'], category)

        preprocess_directory(input_dir, output_dir)
