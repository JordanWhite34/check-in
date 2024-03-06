# preprocessing.py
import os
import cv2
import numpy as np
import shutil
import json
from sklearn.model_selection import train_test_split
from config import PREPROCESSING_CONFIG, DATA_PATHS, SPLIT_RATIOS
from tqdm import tqdm


# Helper function to save mean and std values
def save_mean_std(mean, std, filepath="mean_std.json"):
    with open(filepath, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)


# Helper function to load mean and std values
def load_mean_std(filepath="mean_std.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return np.array(data["mean"]), np.array(data["std"])


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image


def transform_image(image, mean, std):
    resized_image = cv2.resize(image, PREPROCESSING_CONFIG['image_size'])
    # Normalize the image
    resized_image = resized_image.astype(np.float32) / 255.0
    normalized_image = (resized_image - mean) / std
    return normalized_image


def calculate_mean_std(directory_path):
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    image_count = 0

    for filename in tqdm(os.listdir(directory_path), desc="Calculating mean and std"):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            pixel_sum += image.sum(axis=(0, 1))
            pixel_sq_sum += (image ** 2).sum(axis=(0, 1))
            image_count += image.shape[0] * image.shape[1]

    mean = pixel_sum / image_count
    std = np.sqrt((pixel_sq_sum / image_count) - (mean ** 2))

    return mean, std


def preprocess_directory(directory_path, output_directory, mean, std):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in tqdm(os.listdir(directory_path), desc=f"Processing {directory_path}"):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory, filename)

            try:
                image = load_image(input_path)
                transformed_image = transform_image(image, mean, std)

                # Convert normalized image back to 0-255 range for saving
                transformed_image = np.clip(transformed_image, 0, 1)
                transformed_image = (transformed_image * 255).astype(np.uint8)

                # Save the image using OpenCV
                cv2.imwrite(output_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def split_data(source, train_dir, val_dir, test_dir, train_size=SPLIT_RATIOS['train_ratio'],
               val_size=SPLIT_RATIOS['val_ratio']):
    files = [f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    train_files, test_files = train_test_split(files, test_size=1 - train_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (train_size + val_size),
                                              random_state=42)

    for file in tqdm(train_files, desc="Copying train files"):
        shutil.copy(os.path.join(source, file), train_dir)
    for file in tqdm(val_files, desc="Copying val files"):
        shutil.copy(os.path.join(source, file), val_dir)
    for file in tqdm(test_files, desc="Copying test files"):
        shutil.copy(os.path.join(source, file), test_dir)


if __name__ == "__main__":
    categories = ['clean', 'messy']

    # Load or calculate mean and std
    mean_std_filepath = os.path.join(DATA_PATHS['processed'], 'mean_std.json')
    if not os.path.exists(mean_std_filepath):
        all_means = []
        all_stds = []
        # Calculate the mean and std for each category using the training set
        for category in categories:
            input_dir = os.path.join(DATA_PATHS['raw'], category)
            mean, std = calculate_mean_std(input_dir)
            all_means.append(mean)
            all_stds.append(std)

        # Calculate the global mean and std across all categories
        global_mean = np.mean(all_means, axis=0)
        global_std = np.mean(all_stds, axis=0)

        # Save the mean and std values for later use
        save_mean_std(global_mean, global_std, filepath=mean_std_filepath)
    else:
        # Load the mean and std values
        global_mean, global_std = load_mean_std(filepath=mean_std_filepath)

    # Update PREPROCESSING_CONFIG with these values
    PREPROCESSING_CONFIG['mean'] = global_mean
    PREPROCESSING_CONFIG['std'] = global_std

    for category in categories:
        input_dir = os.path.join(DATA_PATHS['raw'], category)
        train_dir = os.path.join(DATA_PATHS['processed'], 'train', category)
        val_dir = os.path.join(DATA_PATHS['processed'], 'val', category)
        test_dir = os.path.join(DATA_PATHS['processed'], 'test', category)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Splitting should only be done once, not every time you run the script
        # Uncomment the next line if you need to re-split the dataset
        # split_data(input_dir, train_dir, val_dir, test_dir)

        # Preprocess each split
        preprocess_directory(train_dir, train_dir, PREPROCESSING_CONFIG['mean'], PREPROCESSING_CONFIG['std'])
        preprocess_directory(val_dir, val_dir, PREPROCESSING_CONFIG['mean'], PREPROCESSING_CONFIG['std'])
        preprocess_directory(test_dir, test_dir, PREPROCESSING_CONFIG['mean'], PREPROCESSING_CONFIG['std'])

        print(f"Preprocessing complete for category '{category}'")
