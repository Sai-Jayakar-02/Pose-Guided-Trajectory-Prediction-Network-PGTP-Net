"""
Helper utilities for training and inference.
"""

import os
import random
import numpy as np
import torch
import yaml
import logging
from typing import Dict, Any, Optional
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get compute device.
    
    Args:
        gpu_id: GPU ID to use
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Note: save_checkpoint and load_checkpoint have been moved to checkpoint.py
# These are kept for backward compatibility but delegate to the new module

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict,
    metrics: Dict,
    path: str,
    scheduler: Optional[Any] = None
):
    """
    Save training checkpoint.
    
    Note: This function is deprecated. Use src.utils.checkpoint.save_checkpoint instead.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        config: Model configuration
        metrics: Current metrics
        path: Save path
        scheduler: Learning rate scheduler (optional)
    """
    from .checkpoint import save_checkpoint as _save_checkpoint
    return _save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=metrics,
        config=config,
        filepath=path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load training checkpoint.
    
    Note: This function is deprecated. Use src.utils.checkpoint.load_checkpoint instead.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load to
    
    Returns:
        Checkpoint dictionary
    """
    from .checkpoint import load_checkpoint as _load_checkpoint
    return _load_checkpoint(
        filepath=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# Note: For advanced logging, use src.utils.logging_utils.Logger
# This basic setup_logging is kept for simple use cases

def setup_logging(
    log_dir: str,
    name: str = 'train',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup basic Python logging to file and console.
    
    For advanced logging (TensorBoard, W&B), use src.utils.logging_utils.Logger.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        level: Logging level
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(
        log_dir, 
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current validation score
        
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def print_model_summary(model: torch.nn.Module, input_shapes: Dict):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_shapes: Dictionary of input shapes
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print(f"\nInput shapes:")
    for name, shape in input_shapes.items():
        print(f"  {name}: {shape}")
    print("\nArchitecture:")
    print(model)
    print("="*60 + "\n")
