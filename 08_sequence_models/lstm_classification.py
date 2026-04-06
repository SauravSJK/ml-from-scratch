import torch
import torch.nn as nn


class SequenceClassifierLSTM(nn.Module):
    """
    LSTM architecture designed for discrete sequence classification.
    Processes the full temporal context before routing the final state to a classification head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(SequenceClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        Note: Incorporating lengths allows for `pack_padded_sequence` usage
        in advanced batching scenarios, ignoring padding tokens.
        """
        out, (h_n, c_n) = self.lstm(x)

        # h_n shape: (num_layers, B, hidden_size)
        # We extract the hidden state from the final layer for the entire batch
        final_hidden_state = h_n[-1, :, :]

        logits = self.classifier(final_hidden_state)
        return logits
