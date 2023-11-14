import os
import torch
from PIL import Image
from torchvision.transforms import v2


def compute_mean_std(input_dirs):
    # Assumes input_dirs is a list of directories to compute mean and std over
    channel_sums, channel_squared_sums, num_batches = 0, 0, 0

    for directory in input_dirs:
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)

                # Accumulate the sum and squared sum
                channel_sums += torch.mean(img_tensor, dim=[1, 2])
                channel_squared_sums += torch.mean(img_tensor**2, dim=[1, 2])
                num_batches += 1
            except IOError as e:
                print(f"Error processing image {img_name}: {e}")

    # Compute the mean and std dev
    mean = channel_sums / num_batches
    std = (channel_squared_sums / num_batches - mean ** 2) ** 0.5
    return mean, std


# Assuming 'data_dir' contains 'clean' and 'messy' subdirectories
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, 'data')

# List of subdirectories
input_dirs = [os.path.join(data_dir, 'raw/clean'), os.path.join(data_dir, 'raw/messy')]

# Compute the mean and std dev
mean, std = compute_mean_std(input_dirs)
print(f"Computed Mean: {mean}")
print(f"Computed Std Dev: {std}")

# Proceed to define transformations and preprocessing as before, but now with the computed mean and std
preprocess_transform = v2.Compose([
    v2.Resize((224, 224)),  # Resize to a common size
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # Convert image to tensor
    v2.Normalize(mean=mean.tolist(), std=std.tolist())  # Use computed mean and std
])


def preprocess_images(input_dir, output_dir, transform):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_counter = 0  # Start numbering images from 0

    # Process each class directory ('clean', 'messy')
    for class_dir in ['clean', 'messy']:
        class_input_dir = os.path.join(input_dir, class_dir)
        class_output_dir = os.path.join(output_dir, class_dir)

        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        for img_name in os.listdir(class_input_dir):
            img_path = os.path.join(class_input_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)

                # Ensure the pixel values are between [0, 1] after normalization
                img_tensor = torch.clamp(img_tensor, 0, 1)

                # Convert tensor to PIL image
                img_processed_pil = v2.ToPILImage()(img_tensor)

                # Save the image with a numbered filename
                output_img_path = os.path.join(class_output_dir, f"{image_counter:04d}.png")
                img_processed_pil.save(output_img_path, 'PNG')

                image_counter += 1  # Increment the image counter

            except IOError as e:
                print(f"Error processing image {img_name}: {e}")


# Directory paths setup
raw_data_dir = os.path.join(data_dir, 'raw')  # The directory containing 'clean' and 'messy' subdirectories
processed_data_dir = os.path.join(data_dir, 'processed')  # The directory where processed images will be saved

# Preprocess the images
preprocess_images(raw_data_dir, processed_data_dir, preprocess_transform)