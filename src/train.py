# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from config import MODEL_PARAMS, DATA_PATHS

# Device configuration
device = torch.device('mps')
print("using", device)

# Define a directory for saving checkpoints
CHECKPOINT_DIR = os.path.join('checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Save checkpoint function
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


# Load checkpoint function
def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])


# Load pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Add dropout before the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(num_ftrs, 2)  # 2 classes: clean and messy
)

# Transfer model to MPS device
device = torch.device('mps')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'], weight_decay=1e-4)

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to your preprocessing size
    transforms.ToTensor(),
    # Add any other transformations you defined in preprocessing.py
])

train_dataset = datasets.ImageFolder(root=DATA_PATHS['augmented'] + '/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=True)

val_dataset = datasets.ImageFolder(root=DATA_PATHS['augmented'] + '/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}

# Training loop
for epoch in range(MODEL_PARAMS['num_epochs']):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f'Epoch [{epoch + 1}/{MODEL_PARAMS["num_epochs"]}], Loss: {running_loss / total:.4f}, Accuracy: {100 * correct / total:.2f}%')

    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # Now, when you call save_checkpoint, it will automatically save to the checkpoints directory
    if epoch % 5 == 0:  # Save every 5 epochs, adjust as needed
        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_filename)

print('Finished Training')
