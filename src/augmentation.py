# augmentation.py
import os
import cv2
import numpy as np
from config import DATA_PATHS


def augment_image(image):
    augmented_images = []

    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotation
    angles = [10, 15, -15]
    for angle in angles:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)

    # Random Cropping
    # Assuming you want to crop 90% of the original size
    crop_size = (int(image.shape[1] * 0.9), int(image.shape[0] * 0.9))
    x = np.random.randint(0, image.shape[1] - crop_size[1])
    y = np.random.randint(0, image.shape[0] - crop_size[0])
    cropped = image[y:y+crop_size[0], x:x+crop_size[1]]
    augmented_images.append(cropped)

    # Zooming
    # Zoom by 10-20%
    zoom_factor = 1 + np.random.uniform(0.1, 0.2)
    zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    x = (zoomed.shape[1] - image.shape[1]) // 2
    y = (zoomed.shape[0] - image.shape[0]) // 2
    zoomed = zoomed[y:y+image.shape[0], x:x+image.shape[1]]
    augmented_images.append(zoomed)

    # Brightness Adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:,:,2] = hsv[:,:,2] * brightness_factor
    hsv[:,:,2][hsv[:,:,2]>255] = 255
    bright_adjusted = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    augmented_images.append(bright_adjusted)

    # Color Jittering
    contrast_factor = np.random.uniform(0.8, 1.2)
    jittered = cv2.addWeighted(image, contrast_factor, np.zeros(image.shape, image.dtype), 0, 0)
    augmented_images.append(jittered)

    return augmented_images


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
    for category in ['test', 'train', 'val']:
        for subcategory in ['clean', 'messy']:
            input_dir = os.path.join(DATA_PATHS['processed'], category, subcategory)
            output_dir = os.path.join(DATA_PATHS['augmented'], category, subcategory)

            process_directory(input_dir, output_dir)

    print('augmentation completed')
