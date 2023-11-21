# demo_preprocessing_augmentation.py
import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import load_image, transform_image
from augmentation import augment_image
from config import DATA_PATHS
import numpy as np


def display_image(image, title="Image"):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    plt.imshow(image)  # Assuming the image is already in RGB format
    plt.title(title)
    plt.axis('off')
    plt.show()


def demo_preprocessing(input_path):
    image = load_image(input_path)
    processed_image = transform_image(image)
    display_image(processed_image, "Processed Image")


def demo_augmentation(input_path):
    """ Demonstrates augmentation on a single image """
    image = load_image(input_path)
    augmented_images = augment_image(image)

    for idx, aug_image in enumerate(augmented_images):
        display_image(aug_image, f"Augmented Image {idx+1}")


if __name__ == "__main__":
    sample_image_path = os.path.join(DATA_PATHS['raw'], 'clean', 'Screenshot 2023-07-17 at 3.16.51 PM.png')  # Update with an actual path

    print("Demonstrating Preprocessing...")
    demo_preprocessing(sample_image_path)

    print("Demonstrating Augmentation...")
    demo_augmentation(sample_image_path)