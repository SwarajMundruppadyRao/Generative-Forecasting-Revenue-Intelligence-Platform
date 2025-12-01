"""
Data preprocessing and feature engineering pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

from utils.config import (
    TRAIN_FILE, TEST_FILE, STORES_FILE, FEATURES_FILE,
    FEATURE_CONFIG, BASE_DIR
)
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "preprocessing.log")


class DataPreprocessor:
    """Preprocess Walmart sales data"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all raw data files"""
        logger.info("Loading raw data files...")
        
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE)
        stores = pd.read_csv(STORES_FILE)
        features = pd.read_csv(FEATURES_FILE)
        
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"Stores shape: {stores.shape}")
        logger.info(f"Features shape: {features.shape}")
        
        return train, test, stores, features
    
    def merge_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        stores: pd.DataFrame,
        features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge all datasets"""
        logger.info("Merging datasets...")
        
        # Merge train with stores
        train = train.merge(stores, on='Store', how='left')
        test = test.merge(stores, on='Store', how='left')
        
        # Merge with features
        train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        
        logger.info(f"Merged train shape: {train.shape}")
        logger.info(f"Merged test shape: {test.shape}")
        
        return train, test
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("Handling missing values...")
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and extract date features"""
        logger.info("Parsing dates...")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for time features
        if FEATURE_CONFIG['cyclical_features']:
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
            df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Weekly_Sales') -> pd.DataFrame:
        """Create lag features"""
        if target_col not in df.columns:
            logger.warning(f"{target_col} not in dataframe, skipping lag features")
            return df
        
        logger.info("Creating lag features...")
        
        # Sort by store, dept, and date
        group_cols = ['Store', 'Dept'] if 'Dept' in df.columns else ['Store']
        df = df.sort_values(group_cols + ['Date'])
        
        for lag in FEATURE_CONFIG['lag_features']:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'Weekly_Sales') -> pd.DataFrame:
        """Create rolling window features"""
        if target_col not in df.columns:
            logger.warning(f"{target_col} not in dataframe, skipping rolling features")
            return df
        
        logger.info("Creating rolling features...")
        
        group_cols = ['Store', 'Dept'] if 'Dept' in df.columns else ['Store']
        df = df.sort_values(group_cols + ['Date'])
        
        for window in FEATURE_CONFIG['rolling_windows']:
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        categorical_cols = ['Type'] if 'Type' in df.columns else []
        
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling features...")
        
        # Identify numeric columns to scale
        # Explicitly exclude Weekly_Sales from scaling as it's the target
        exclude_cols = ['Store', 'Dept', 'Date', 'Year', 'Month', 'Week', 'DayOfWeek', 
                       'DayOfMonth', 'Quarter', 'IsHoliday', 'Weekly_Sales']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Filter to only columns present in the dataframe
        scale_cols = [col for col in scale_cols if col in df.columns]
        
        if not scale_cols:
            return df
            
        if fit:
            self.scalers['features'] = StandardScaler()
            df[scale_cols] = self.scalers['features'].fit_transform(df[scale_cols])
        else:
            if 'features' in self.scalers:
                # Only transform columns that were seen during fit
                # This handles cases where test might have different columns (though shouldn't happen here)
                try:
                    df[scale_cols] = self.scalers['features'].transform(df[scale_cols])
                except ValueError as e:
                    logger.warning(f"Scaling mismatch: {e}. Skipping scaling for test set.")
        
        return df
    
    def preprocess_pipeline(
        self,
        save_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline
        
        Returns:
            Processed train and test dataframes
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        train, test, stores, features = self.load_raw_data()
        
        # Merge datasets
        train, test = self.merge_data(train, test, stores, features)
        
        # Parse dates
        train = self.parse_dates(train)
        test = self.parse_dates(test)
        
        # Handle missing values
        train = self.handle_missing_values(train)
        test = self.handle_missing_values(test)
        
        # Create lag features (only for train initially)
        train = self.create_lag_features(train)
        
        # Create rolling features
        train = self.create_rolling_features(train)
        
        # Encode categorical
        train = self.encode_categorical(train, fit=True)
        test = self.encode_categorical(test, fit=False)
        
        # For test, we need to handle lag features differently
        # Combine train and test temporarily for lag feature creation
        test['Weekly_Sales'] = np.nan  # Placeholder
        combined = pd.concat([train, test], ignore_index=True)
        combined = combined.sort_values(['Store', 'Dept', 'Date'] if 'Dept' in combined.columns else ['Store', 'Date'])
        
        # Recreate lag and rolling features on combined data
        combined = self.create_lag_features(combined)
        combined = self.create_rolling_features(combined)
        
        # Split back
        train = combined[combined['Weekly_Sales'].notna()].copy()
        test = combined[combined['Weekly_Sales'].isna()].copy()
        test = test.drop('Weekly_Sales', axis=1)
        
        # Scale features
        train = self.scale_features(train, fit=True)
        test = self.scale_features(test, fit=False)
        
        # Store feature columns
        self.feature_columns = [col for col in train.columns 
                               if col not in ['Weekly_Sales', 'Date']]
        
        logger.info("Preprocessing complete!")
        logger.info(f"Final train shape: {train.shape}")
        logger.info(f"Final test shape: {test.shape}")
        
        # Save preprocessed data and preprocessor
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            train.to_csv(save_path / 'train_processed.csv', index=False)
            test.to_csv(save_path / 'test_processed.csv', index=False)
            
            # Save preprocessor
            with open(save_path / 'preprocessor.pkl', 'wb') as f:
                pickle.dump({
                    'scalers': self.scalers,
                    'encoders': self.encoders,
                    'feature_columns': self.feature_columns
                }, f)
            
            logger.info(f"Saved preprocessed data to {save_path}")
        
        return train, test
    
    def load_preprocessor(self, path: Path):
        """Load saved preprocessor"""
        with open(path / 'preprocessor.pkl', 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.encoders = data['encoders']
            self.feature_columns = data['feature_columns']
        logger.info("Loaded preprocessor")


def main():
    """Run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    train, test = preprocessor.preprocess_pipeline(
        save_path=BASE_DIR / 'data' / 'processed'
    )
    print(f"Preprocessing complete!")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Feature columns: {len(preprocessor.feature_columns)}")


if __name__ == "__main__":
    main()
