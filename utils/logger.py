"""
Logging utilities for the platform
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger"""
    return logging.getLogger(name)


class MetricsLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        # Initialize metrics file
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("timestamp,epoch,phase,loss,rmse,mae,mape\n")
    
    def log_metrics(self, epoch: int, phase: str, metrics: dict):
        """Log metrics to file"""
        timestamp = datetime.now().isoformat()
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{epoch},{phase},{metrics.get('loss', 0):.6f},"
                   f"{metrics.get('rmse', 0):.6f},{metrics.get('mae', 0):.6f},"
                   f"{metrics.get('mape', 0):.6f}\n")
    
    def log_message(self, message: str):
        """Log a message"""
        logger = get_logger(self.experiment_name)
        logger.info(message)
