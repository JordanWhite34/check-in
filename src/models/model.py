import torch.nn as nn
import torch.nn.functional as F

# This is not being used anymore


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # With 224x224 input, conv2 output will be 64 x 56 x 56
        self.fc1 = nn.Linear(64 * 56 * 56, 512)

        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(512, 2)

        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten conv output to feed into fc1
        x = x.view(-1, 64 * 56 * 56)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x