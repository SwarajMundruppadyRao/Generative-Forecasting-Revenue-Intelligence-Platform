"""
Configuration management for the Forecasting Platform
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env' if 'BASE_DIR' in dir() else Path(__file__).parent.parent / '.env')

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "forecasting" / "models"
FAISS_DIR = BASE_DIR / "rag" / "faiss_index"
CORPUS_DIR = BASE_DIR / "rag" / "corpus"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
STORES_FILE = DATA_DIR / "stores.csv"
FEATURES_FILE = DATA_DIR / "features.csv"

# Model configurations
LSTM_CONFIG = {
    "input_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True
}

TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_length": 52  # 52 weeks
}

# GNN Configuration
GNN_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "heads": 4,  # for GAT
    "model_type": "GAT"  # or "GraphSAGE"
}

# Training configurations
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "sequence_length": 12,  # 12 weeks lookback
    "forecast_horizon": 4,  # 4 weeks ahead
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu"
}

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": "gpt-4o-mini",  # Cheapest model: $0.15/1M input tokens
    "temperature": 0.7,
    "max_tokens": 1000,
    "embedding_model": "text-embedding-3-small"
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "dense_weight": 0.5,
    "sparse_weight": 0.3,
    "gnn_weight": 0.2,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384
}

# FastAPI Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "title": "Revenue Intelligence Platform",
    "version": "1.0.0"
}

# Feature Engineering
FEATURE_CONFIG = {
    "lag_features": [1, 2, 3, 4, 7, 14, 28],
    "rolling_windows": [7, 14, 28],
    "holiday_features": True,
    "cyclical_features": True
}


def get_config(config_name: str) -> Dict[str, Any]:
    """Get configuration by name"""
    configs = {
        "lstm": LSTM_CONFIG,
        "transformer": TRANSFORMER_CONFIG,
        "gnn": GNN_CONFIG,
        "training": TRAINING_CONFIG,
        "neo4j": NEO4J_CONFIG,
        "openai": OPENAI_CONFIG,
        "rag": RAG_CONFIG,
        "api": API_CONFIG,
        "features": FEATURE_CONFIG
    }
    return configs.get(config_name, {})


def save_config(config: Dict[str, Any], filepath: Path):
    """Save configuration to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
