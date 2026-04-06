import torch
import torch.nn as nn

class MNIST_MLP(nn.Module):
    """
    A robust Multi-Layer Perceptron for MNIST Classification.
    Includes standard regularization techniques (Dropout, BatchNorm) for stable training.
    """
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10, dropout_rate: float = 0.5):
        super(MNIST_MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer (No Softmax here; handled by CrossEntropyLoss in train.py)
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation. 
        Note: The input shape should be (B, C, H, W) or (B, Features).
        """
        return self.network(x)