import torch
from torch.utils.data import Dataset, DataLoader


class CustomTensorDataset(Dataset):
    """
    A production-ready Dataset wrapper for custom tensors.
    Extracted from custom deep learning workflows to handle raw array ingestion.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.size(0) == labels.size(0), (
            "Features and labels must have the same number of samples."
        )
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """
    Utility function to generate train and validation dataloaders.
    """
    train_dataset = CustomTensorDataset(X_train, y_train)
    val_dataset = CustomTensorDataset(X_val, y_val)

    # Pin memory for faster CPU to GPU/MPS data transfer
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader
