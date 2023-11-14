# config.py

# Data paths (relative to project root directory)
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'augmented': 'data/augmented'
}

# Image Preprocessing Settings
PREPROCESSING_CONFIG = {
    'image_size': (224, 224),  # Common size for CNNs
    'mean': [0.485, 0.456, 0.406],  # Mean for normalization (ImageNet standards)
    'std': [0.229, 0.224, 0.225]  # Std for normalization (ImageNet standards)
}

# Model Hyperparameters
MODEL_PARAMS = {
    'learning_rate': 1e-4,  # A good starting point for many models
    'batch_size': 32,  # Adjust based on memory availability
    'num_epochs': 25,  # Initial epochs count
}

# Training Settings
TRAINING_CONFIG = {
    'optimizer': 'adam',  # Commonly used optimizer
    'loss_function': 'cross_entropy',  # Suitable for classification tasks
    'scheduler_step_size': 7,  # Step size for learning rate decay
    'scheduler_gamma': 0.1,  # Decay factor for learning rate
}

# Additional settings can be added as needed
