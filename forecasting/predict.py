"""
Prediction pipeline for forecasting models
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from forecasting.model_lstm import create_lstm_model
from forecasting.model_transformer import create_transformer_model
from forecasting.dataset import TimeSeriesDataset
from utils.config import TRAINING_CONFIG, MODELS_DIR, BASE_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "prediction.log")


class ForecastingPredictor:
    """Predictor for forecasting models"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        model_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize predictor
        
        Args:
            model_type: 'lstm' or 'transformer'
            model_path: Path to model checkpoint
            device: Device to use
        """
        self.model_type = model_type
        self.device = device or TRAINING_CONFIG['device']
        
        # Load model
        if model_path is None:
            model_path = MODELS_DIR / f"{model_type}_forecaster_best.pth"
        
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded {model_type} model from {model_path}")
    
    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Infer input size from checkpoint
        if self.model_type == "lstm":
            # Get input size from first layer
            input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
            model = create_lstm_model(input_size)
        else:
            # Get input size from input projection
            input_size = checkpoint['model_state_dict']['input_projection.weight'].shape[1]
            model = create_transformer_model(input_size)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict(
        self,
        features: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            features: Input features of shape (batch_size, seq_length, n_features)
                     or (seq_length, n_features) for single sample
        
        Returns:
            Predictions of shape (batch_size, forecast_horizon) or (forecast_horizon,)
        """
        # Convert to tensor
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        features = features.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features)
        
        predictions = predictions.cpu().numpy()
        
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    def predict_store(
        self,
        data: pd.DataFrame,
        store_id: int,
        dept_id: Optional[int] = None,
        horizon: int = 4
    ) -> Dict:
        """
        Predict for a specific store/department
        
        Args:
            data: Historical data
            store_id: Store ID
            dept_id: Department ID (optional)
            horizon: Forecast horizon
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Filter data
        mask = data['Store'] == store_id
        if dept_id is not None:
            mask &= data['Dept'] == dept_id
        
        store_data = data[mask].copy()
        
        if len(store_data) == 0:
            logger.warning(f"No data found for Store {store_id}, Dept {dept_id}")
            return {
                'store_id': store_id,
                'dept_id': dept_id,
                'predictions': [],
                'error': 'No data found'
            }
        
        # Sort by date
        store_data = store_data.sort_values('Date')
        
        # Get last sequence_length records
        seq_length = TRAINING_CONFIG['sequence_length']
        if len(store_data) < seq_length:
            logger.warning(f"Insufficient data for Store {store_id}, Dept {dept_id}")
            return {
                'store_id': store_id,
                'dept_id': dept_id,
                'predictions': [],
                'error': 'Insufficient data'
            }
        
        # Prepare features - only use numeric columns
        # Get all numeric columns
        numeric_cols = store_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Weekly_Sales', 'Store', 'Dept']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Fill NaN values
        store_data[feature_cols] = store_data[feature_cols].fillna(0)
        
        features = store_data[feature_cols].iloc[-seq_length:].values
        
        # Predict
        predictions = self.predict(features)
        
        # Get last date
        last_date = pd.to_datetime(store_data['Date'].iloc[-1])
        
        # Generate future dates (assuming weekly data)
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(horizon)]
        
        result = {
            'store_id': store_id,
            'dept_id': dept_id,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'predictions': predictions[:horizon].tolist(),
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'model_type': self.model_type
        }
        
        return result
    
    def batch_predict(
        self,
        data: pd.DataFrame,
        store_ids: Optional[List[int]] = None,
        horizon: int = 4
    ) -> List[Dict]:
        """
        Batch predictions for multiple stores
        
        Args:
            data: Historical data
            store_ids: List of store IDs (if None, predict for all)
            horizon: Forecast horizon
        
        Returns:
            List of prediction dictionaries
        """
        if store_ids is None:
            store_ids = data['Store'].unique()
        
        results = []
        
        for store_id in store_ids:
            # Get departments for this store
            store_data = data[data['Store'] == store_id]
            
            if 'Dept' in store_data.columns:
                dept_ids = store_data['Dept'].unique()
                
                for dept_id in dept_ids:
                    result = self.predict_store(data, store_id, dept_id, horizon)
                    results.append(result)
            else:
                result = self.predict_store(data, store_id, None, horizon)
                results.append(result)
        
        logger.info(f"Generated predictions for {len(results)} store-department combinations")
        
        return results


def main():
    """Main prediction script"""
    # Load preprocessed data
    processed_dir = BASE_DIR / 'data' / 'processed'
    data = pd.read_csv(processed_dir / 'train_processed.csv')
    
    # Create predictors
    lstm_predictor = ForecastingPredictor(model_type="lstm")
    transformer_predictor = ForecastingPredictor(model_type="transformer")
    
    # Example: Predict for store 1
    logger.info("Making predictions for Store 1...")
    
    lstm_result = lstm_predictor.predict_store(data, store_id=1, dept_id=1)
    transformer_result = transformer_predictor.predict_store(data, store_id=1, dept_id=1)
    
    print("\nLSTM Predictions:")
    print(f"Store: {lstm_result['store_id']}, Dept: {lstm_result['dept_id']}")
    print(f"Predictions: {lstm_result['predictions']}")
    
    print("\nTransformer Predictions:")
    print(f"Store: {transformer_result['store_id']}, Dept: {transformer_result['dept_id']}")
    print(f"Predictions: {transformer_result['predictions']}")


if __name__ == "__main__":
    main()
