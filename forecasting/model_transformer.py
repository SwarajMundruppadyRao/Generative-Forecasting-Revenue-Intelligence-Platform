"""
Transformer-based forecasting model
"""
import torch
import torch.nn as nn
import math
from typing import Optional

from utils.config import TRANSFORMER_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        forecast_horizon: int = 4,
        max_seq_length: int = 52
    ):
        """
        Initialize Transformer forecaster
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            forecast_horizon: Number of steps to forecast
            max_seq_length: Maximum sequence length
        """
        super(TransformerForecaster, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            src_mask: Optional source mask
        
        Returns:
            Output tensor of shape (batch_size, forecast_horizon)
        """
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_mask)
        
        # Use mean pooling over sequence
        pooled = encoded.mean(dim=1)
        
        # Forecast
        output = self.fc(pooled)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def create_transformer_model(input_size: int, config: dict = None) -> TransformerForecaster:
    """
    Create Transformer model with config
    
    Args:
        input_size: Input feature dimension
        config: Model configuration
    
    Returns:
        Transformer model
    """
    if config is None:
        config = TRANSFORMER_CONFIG.copy()
    
    config['input_size'] = input_size
    
    model = TransformerForecaster(**config)
    
    return model
