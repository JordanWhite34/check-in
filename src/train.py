# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import MODEL_PARAMS, DATA_PATHS


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    CHECKPOINT_DIR = os.path.join('checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename, map_location='mps')  # Adjust map_location based on your setup
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])


def get_transforms():
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add any other transformations here
    ])
    return transform


def get_dataloaders(batch_size, train_dir, val_dir, transform):
    # Assuming DATA_PATHS dict or similar setup for paths
    train_dataset = datasets.ImageFolder(root=DATA_PATHS['augmented'] + '/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=True)

    val_dataset = datasets.ImageFolder(root=DATA_PATHS['augmented'] + '/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=MODEL_PARAMS['batch_size'], shuffle=False)

    return train_loader, val_loader


def train(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs, patience=5):
    best_val_loss = float('inf')
    no_improve_epochs = 0  # Counter for early stopping
    best_val_accuracy = 0  # Variable to keep track of the best validation accuracy

    # Training loop
    for epoch in range(MODEL_PARAMS['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{MODEL_PARAMS["num_epochs"]}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

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

        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_filename)

        # # Save checkpoint if current validation accuracy is higher than the best seen so far
        # if val_accuracy > best_val_accuracy:
        #     print(f'New best validation accuracy: {val_accuracy:.2f}%, saving checkpoint...')
        #     best_val_accuracy = val_accuracy
        #     checkpoint_filename = f"checkpoint_epoch_{epoch}_val_acc_{val_accuracy:.2f}.pth.tar"
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, filename=checkpoint_filename)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            # Save checkpoint as this is the best model so far
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename="best_checkpoint")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs. Early stopping...")
                break

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

    print('Finished Training')


if __name__ == "__main__":
    # Device configuration
    device = torch.device('mps')
    print("using", device)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Modify model as needed
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'], weight_decay=MODEL_PARAMS['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    transform = get_transforms()
    train_loader, val_loader = get_dataloaders(MODEL_PARAMS['batch_size'], DATA_PATHS['augmented'] + '/train', DATA_PATHS['augmented'] + '/val', transform)
    dataloaders = {'train': train_loader, 'val': val_loader}

    train(model, dataloaders, device, criterion, optimizer, scheduler, MODEL_PARAMS['num_epochs'])
