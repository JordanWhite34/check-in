# utils.py
import torch


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the model and training parameters at the specified filename. Call during training to save the best model."""
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint."""
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def print_overwrite(step, total_step, loss, operation):
    """Prints over the same line to give a dynamic loading effect. Great for monitoring training and evaluation operations."""
    if operation == 'train':
        print('Training Step: {}/{} Loss: {:.4f}'.format(step, total_step, loss), end='\r')
    elif operation == 'valid':
        print('Validation Step: {}/{} Loss: {:.4f}'.format(step, total_step, loss), end='\r')
