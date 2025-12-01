"""
Training pipeline for forecasting models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm

from forecasting.dataset import create_dataloaders, split_train_val
from forecasting.model_lstm import create_lstm_model
from forecasting.model_transformer import create_transformer_model
from utils.config import TRAINING_CONFIG, MODELS_DIR, BASE_DIR
from utils.logger import setup_logger, MetricsLogger

logger = setup_logger(__name__, BASE_DIR / "logs" / "training.log")


class ForecastingTrainer:
    """Trainer for forecasting models"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[str] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            model_name: Name for saving model
            device: Device to use
        """
        self.model = model
        self.model_name = model_name
        self.device = device or TRAINING_CONFIG['device']
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Metrics logger
        self.metrics_logger = MetricsLogger(
            BASE_DIR / "logs",
            experiment_name=model_name
        )
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # MAPE
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        for features, targets in pbar:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        num_epochs = num_epochs or TRAINING_CONFIG['num_epochs']
        early_stopping_patience = early_stopping_patience or TRAINING_CONFIG['early_stopping_patience']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"RMSE: {train_metrics['rmse']:.4f}, "
                       f"MAE: {train_metrics['mae']:.4f}, "
                       f"MAPE: {train_metrics['mape']:.2f}%")
            
            self.metrics_logger.log_metrics(epoch, 'train', train_metrics)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                           f"RMSE: {val_metrics['rmse']:.4f}, "
                           f"MAE: {val_metrics['mae']:.4f}, "
                           f"MAPE: {val_metrics['mape']:.2f}%")
                
                self.metrics_logger.log_metrics(epoch, 'val', val_metrics)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_model(is_best=True)
                    logger.info("Saved best model")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(is_best=False, epoch=epoch)
        
        logger.info("Training complete!")
    
    def save_model(self, is_best: bool = False, epoch: Optional[int] = None):
        """Save model checkpoint"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            path = MODELS_DIR / f"{self.model_name}_best.pth"
        elif epoch is not None:
            path = MODELS_DIR / f"{self.model_name}_epoch_{epoch}.pth"
        else:
            path = MODELS_DIR / f"{self.model_name}_latest.pth"
        
        torch.save(checkpoint, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded model from {path}")


def train_lstm_model(train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
    """Train LSTM model"""
    logger.info("Training LSTM model...")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_data, val_data)
    
    # Get input size from first batch
    sample_features, _ = next(iter(train_loader))
    input_size = sample_features.shape[-1]
    
    # Create model
    model = create_lstm_model(input_size)
    logger.info(f"Created LSTM model with input size {input_size}")
    
    # Train
    trainer = ForecastingTrainer(model, "lstm_forecaster")
    trainer.train(train_loader, val_loader)
    
    return trainer


def train_transformer_model(train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
    """Train Transformer model"""
    logger.info("Training Transformer model...")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_data, val_data)
    
    # Get input size from first batch
    sample_features, _ = next(iter(train_loader))
    input_size = sample_features.shape[-1]
    
    # Create model
    model = create_transformer_model(input_size)
    logger.info(f"Created Transformer model with input size {input_size}")
    
    # Train
    trainer = ForecastingTrainer(model, "transformer_forecaster")
    trainer.train(train_loader, val_loader)
    
    return trainer


def main():
    """Main training script"""
    # Load preprocessed data
    processed_dir = BASE_DIR / 'data' / 'processed'
    train_data = pd.read_csv(processed_dir / 'train_processed.csv')
    
    # Split train/val
    train_data, val_data = split_train_val(train_data, val_split=0.2)
    
    # Train both models
    logger.info("=" * 50)
    logger.info("Training LSTM Model")
    logger.info("=" * 50)
    train_lstm_model(train_data, val_data)
    
    logger.info("\n" + "=" * 50)
    logger.info("Training Transformer Model")
    logger.info("=" * 50)
    train_transformer_model(train_data, val_data)


if __name__ == "__main__":
    main()
