"""
LSTM-based forecasting model
"""
import torch
import torch.nn as nn
from typing import Tuple

from utils.config import LSTM_CONFIG


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        forecast_horizon: int = 4
    ):
        """
        Initialize LSTM forecaster
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            forecast_horizon: Number of steps to forecast
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, forecast_horizon)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Forecast
        output = self.fc(hidden)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def create_lstm_model(input_size: int, config: dict = None) -> LSTMForecaster:
    """
    Create LSTM model with config
    
    Args:
        input_size: Input feature dimension
        config: Model configuration
    
    Returns:
        LSTM model
    """
    if config is None:
        config = LSTM_CONFIG.copy()
    
    config['input_size'] = input_size
    
    model = LSTMForecaster(**config)
    
    return model
