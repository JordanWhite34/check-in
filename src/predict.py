import torch
from PIL import Image
from torchvision.transforms import Compose
from config import MODEL_PARAMS, DATA_PATHS
from train import get_transforms  # Assuming get_transforms provides necessary transformations
from evaluate import initialize_model, load_checkpoint  # Ensure these are correctly imported


def predict_single_image(image_path, model, device, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')
    # Apply transformations
    transformed_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Perform prediction
    model.eval()
    with torch.no_grad():
        outputs = model(transformed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1).values.item()

    # Map predicted class to 'clean' or 'messy'
    class_label = 'messy' if predicted_class == 1 else 'clean'

    return class_label, confidence


# Main function for demonstration
if __name__ == '__main__':
    device = torch.device('mps')
    model = initialize_model().to(device)
    checkpoint_path = 'checkpoints/checkpoint_epoch_15.pth.tar'  # Adjust path as needed
    load_checkpoint(checkpoint_path, model, device)

    # Path to your single image
    single_image_path = '/Users/jordanwhite/Downloads/messy.png'
    transform = get_transforms()  # Ensure this returns a torchvision.transforms.Compose object suitable for model

    predicted_class, confidence = predict_single_image(single_image_path, model, device, transform)
    print(f'Predicted Class: {predicted_class}, Confidence: {100*confidence:.2f}%')
