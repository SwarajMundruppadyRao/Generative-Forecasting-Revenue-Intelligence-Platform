"""
PyTorch Dataset for time series forecasting
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from utils.config import TRAINING_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesDataset(Dataset):
    """Time series dataset for forecasting"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        forecast_horizon: int,
        target_col: str = 'Weekly_Sales',
        feature_cols: Optional[List[str]] = None,
        group_cols: Optional[List[str]] = None
    ):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with time series data
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            target_col: Target column name
            feature_cols: List of feature column names
            group_cols: Columns to group by (e.g., ['Store', 'Dept'])
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.group_cols = group_cols or ['Store', 'Dept']
        
        # Determine feature columns - only use numeric columns
        if feature_cols is None:
            # Get all numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = [target_col] + self.group_cols
            self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        # Fill any remaining NaN values
        self.data[self.feature_cols] = self.data[self.feature_cols].fillna(0)
        if target_col in self.data.columns:
            self.data[target_col] = self.data[target_col].fillna(0)
        
        # Sort data
        if 'Date' in self.data.columns:
            self.data = self.data.sort_values(self.group_cols + ['Date'])
        else:
            self.data = self.data.sort_values(self.group_cols)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
        logger.info(f"Feature dimension: {len(self.feature_cols)}")
    
    def _create_sequences(self) -> List[Tuple]:
        """Create sequences from data"""
        sequences = []
        
        # Group by store/dept
        for group_key, group_df in self.data.groupby(self.group_cols):
            group_df = group_df.sort_values('Date').reset_index(drop=True)
            
            # Extract features and target
            features = group_df[self.feature_cols].values
            target = group_df[self.target_col].values
            
            # Create sequences
            for i in range(len(group_df) - self.sequence_length - self.forecast_horizon + 1):
                seq_features = features[i:i + self.sequence_length]
                seq_target = target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                
                # Store group info for later use
                sequences.append({
                    'features': seq_features,
                    'target': seq_target,
                    'group_key': group_key,
                    'start_idx': i
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        seq = self.sequences[idx]
        
        features = torch.FloatTensor(seq['features'])
        target = torch.FloatTensor(seq['target'])
        
        return features, target
    
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return len(self.feature_cols)


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders
    
    Args:
        train_data: Training dataframe
        val_data: Validation dataframe
        batch_size: Batch size
        sequence_length: Sequence length
        forecast_horizon: Forecast horizon
        **kwargs: Additional arguments for TimeSeriesDataset
    
    Returns:
        Train and validation dataloaders
    """
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    sequence_length = sequence_length or TRAINING_CONFIG['sequence_length']
    forecast_horizon = forecast_horizon or TRAINING_CONFIG['forecast_horizon']
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Set to False for Windows compatibility
    )
    
    val_loader = None
    if val_data is not None:
        val_dataset = TimeSeriesDataset(
            val_data,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            **kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False  # Set to False for Windows compatibility
        )
    
    return train_loader, val_loader


def split_train_val(
    data: pd.DataFrame,
    val_split: float = 0.2,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation
    
    Args:
        data: Input dataframe
        val_split: Validation split ratio
        time_based: If True, split by time; otherwise random
    
    Returns:
        Train and validation dataframes
    """
    if time_based:
        # Sort by date and split
        data = data.sort_values('Date')
        split_idx = int(len(data) * (1 - val_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
    else:
        # Random split
        train_data = data.sample(frac=1 - val_split, random_state=42)
        val_data = data.drop(train_data.index)
    
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    return train_data, val_data
