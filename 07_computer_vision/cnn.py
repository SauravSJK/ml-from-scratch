import torch
import torch.nn as nn

class MNIST_CNN(nn.Module):
    """
    A robust Convolutional Neural Network for image classification.
    Includes explicit tracking of spatial dimensions (N, C, H, W).
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(MNIST_CNN, self).__init__()
        
        # Block 1: Feature Extraction
        # Input: (B, 1, 28, 28)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # -> (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 32, 14, 14)
        )
        
        # Block 2: Deep Feature Extraction
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # -> (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 64, 7, 7)
        )
        
        # Block 3: Classification Head
        # Flattened size: 64 channels * 7 height * 7 width = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass defining the spatial reduction pipeline.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x