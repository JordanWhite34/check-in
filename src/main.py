import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from preprocessing import (
    preprocess_directory,
    split_data,
)
from train import get_transforms, get_dataloaders, train
from evaluate import load_checkpoint, get_evaluation_dataloader, evaluate

from config import MODEL_PARAMS, DATA_PATHS


def main():
    # preprocessing data
    categories = ['clean', 'messy']
    for category in categories:
        input_dir = os.path.join(DATA_PATHS['raw'], category)
        train_dir = os.path.join(DATA_PATHS['processed'], 'train', category)
        val_dir = os.path.join(DATA_PATHS['processed'], 'val', category)
        test_dir = os.path.join(DATA_PATHS['processed'], 'test', category)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        split_data(input_dir, train_dir, val_dir, test_dir)

        # Now preprocess each split
        preprocess_directory(train_dir, train_dir)
        preprocess_directory(val_dir, val_dir)
        preprocess_directory(test_dir, test_dir)

    # Training Model
    # Device configuration
    device = torch.device('mps')
    print("using", device)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Modify model as needed
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'],
                           weight_decay=MODEL_PARAMS['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    transform = get_transforms()
    train_loader, val_loader = get_dataloaders(MODEL_PARAMS['batch_size'], DATA_PATHS['augmented'] + '/train',
                                               DATA_PATHS['augmented'] + '/val', transform)
    dataloaders = {'train': train_loader, 'val': val_loader}

    train(model, dataloaders, device, criterion, optimizer, scheduler, MODEL_PARAMS['num_epochs'])

    # Evaluate Model
    checkpoint_path = 'checkpoints/best_checkpoint.pth.tar'  # Adjust path as needed
    load_checkpoint(checkpoint_path, model, device)

    # Setup DataLoader
    batch_size = MODEL_PARAMS['batch_size']
    data_dir = DATA_PATHS['augmented'] + '/test'
    transform = get_transforms()
    evaluation_loader = get_evaluation_dataloader(batch_size, data_dir, transform)

    # Perform evaluation
    evaluate(model, evaluation_loader, device)


if __name__ == "__main__":
    main()
