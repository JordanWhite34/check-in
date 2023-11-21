# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MyModel  # Replace with your actual model import
from config import MODEL_PARAMS, DATA_PATHS
from utils import save_checkpoint, load_checkpoint  # Assuming you have these utility functions

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloaders, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_accuracy = correct_preds / total_preds
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_correct_preds = 0
            val_total_preds = 0

            for images, labels in dataloaders['val']:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct_preds += (predicted == labels).sum().item()
                val_total_preds += labels.size(0)

            val_loss = val_running_loss / len(dataloaders['val'].dataset)
            val_accuracy = val_correct_preds / val_total_preds
            print(f'Validation Loss: {val_loss}, Accuracy: {val_accuracy}')

            # Save the model if validation accuracy has increased
            if val_accuracy > best_accuracy:
                print('Validation Accuracy Improved from {:.4f} to {:.4f}'.format(best_accuracy, val_accuracy))
                best_accuracy = val_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                }
                save_checkpoint(checkpoint, filename="best_checkpoint.pth.tar")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example, adjust to your needs
        transforms.ToTensor(),
        # Add any other transformations from your preprocessing.py
    ])

    # Load Datasets
    train_dataset = datasets.ImageFolder(root=DATA_PATHS['processed'] + '/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=DATA_PATHS['processed'] + '/val', transform=transform)

    # Data Loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=True),
        'val': DataLoader(val_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=False)
    }

    # Model, Loss Function, Optimizer
    model = MyModel()  # Replace with your actual model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'])

    # Number of epochs
    num_epochs = MODEL_PARAMS['num_epochs']

    # Start Training
    train(model, dataloaders, criterion, optimizer, num_epochs, device)
