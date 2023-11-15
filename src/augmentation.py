# augmentation.py
import os
import cv2
import numpy as np
from config import DATA_PATHS

def augment_image(image):
    # Define augmentation transformations here
    # Example: Flip, rotate, adjust brightness, etc.

    # Horizontal flip
    flipped = cv2.flip(image, 1)

    # Rotation
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=15, scale=1)
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Brightness adjustment (increase by 10%)
    brightness_adjusted = cv2.convertScaleAbs(image, alpha=1.1, beta=0)

    return [flipped, rotated, brightness_adjusted]

def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_directory, filename)

            try:
                # Load image
                image = cv2.imread(input_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Generate augmented images
                augmented_images = augment_image(image)

                # Save augmented images
                for idx, aug_image in enumerate(augmented_images):
                    output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_aug{idx}.png")
                    aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, aug_image)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    for category in ['clean', 'messy']:
        input_dir = os.path.join(DATA_PATHS['processed'], category)
        output_dir = os.path.join(DATA_PATHS['augmented'], category)

        process_directory(input_dir, output_dir)
