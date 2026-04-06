import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    """
    LSTM architecture designed for continuous time-series forecasting.
    Outputs a sequence of predictions or a single future step based on configuration.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True ensures input tensors are (Batch, Sequence, Feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear regression head to map hidden states back to continuous values
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args: x of shape (B, Seq_Len, Input_Size)
        """
        # Dynamic initialization of hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # out shape: (B, Seq_Len, Hidden_Size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply the regressor to the final sequence step to predict the next value
        # taking out[:, -1, :] grabs the hidden state of the last time step
        predictions = self.regressor(out[:, -1, :]) 
        
        return predictions