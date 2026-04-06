import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    """
    A modular training engine for PyTorch models.
    Handles device placement, loss tracking, and validation loops.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        device: str = None,
    ):

        # Hardware-agnostic device selection (Supporting Apple Silicon)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Standard Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 1. Zero the gradients
            self.optimizer.zero_grad()

            # 2. Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 3. Backward pass & Optimizer step
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient tracking for inference to save memory
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        return val_loss / len(self.val_loader), accuracy

    def fit(self, epochs: int):
        print(f"Starting training on device: {self.device}")
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
