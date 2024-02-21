import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet-50 model
loaded_model = models.resnet50(False)  # Load the base ResNet model without pre-trained weights

# Add dropout before the final fully connected layer
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(num_ftrs, 2)  # 2 classes: clean and messy
)

import torch

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])


# Load the checkpoint
load_checkpoint(loaded_model, optimizer=None, filename="checkpoints/checkpoint_epoch_0.pth.tar")

# Set the model to evaluation mode
loaded_model.eval()

# Now you can use this loaded_model for inference
