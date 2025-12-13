"""
Checkpoint Management Utilities.

Provides comprehensive checkpoint functionality:
- Save/load model checkpoints with metadata
- Best model tracking
- Checkpoint averaging (SWA)
- Model export (ONNX, TorchScript)
- Automatic checkpoint cleanup

Usage:
    from src.utils import CheckpointManager
    
    # Initialize manager
    manager = CheckpointManager(
        checkpoint_dir='checkpoints',
        max_checkpoints=5,
        save_best=True,
        metric_name='val_ade',
        mode='min',
    )
    
    # Save checkpoint
    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics={'val_ade': 0.41, 'val_fde': 0.82},
    )
    
    # Load checkpoint
    checkpoint = manager.load_best()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Resume training
    epoch, metrics = manager.resume(model, optimizer, scheduler)
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
import glob

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Basic Save/Load Functions
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    epoch: int = 0,
    step: int = 0,
    metrics: Dict[str, float] = None,
    config: Dict[str, Any] = None,
    filepath: str = 'checkpoint.pt',
    save_optimizer: bool = True,
    **kwargs,
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: LR scheduler (optional)
        epoch: Current epoch
        step: Current step
        metrics: Metrics dictionary
        config: Model configuration
        filepath: Output path
        save_optimizer: Whether to save optimizer state
        **kwargs: Additional data to save
    
    Returns:
        Path to saved checkpoint
    """
    # Handle DataParallel wrapper
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
        'config': config or {},
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add optimizer state
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add extra data
    checkpoint.update(kwargs)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    logger.info(f"Checkpoint saved: {filepath}")
    
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = 'cuda',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Target device
        strict: Strict state dict loading
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    if model is not None:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            # Handle keys with 'module.' prefix
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=strict)
        
        logger.info(f"Model weights loaded from {filepath}")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state loaded")
    
    return checkpoint


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Manage model checkpoints with best tracking and cleanup.
    
    Features:
    - Save checkpoints at regular intervals
    - Track best model based on metric
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoint
    - Export to ONNX/TorchScript
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        filename_template: str = 'checkpoint_epoch_{epoch:03d}.pt',
        best_filename: str = 'best.pt',
        last_filename: str = 'last.pt',
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_best: Whether to track best model
            metric_name: Metric for best model selection
            mode: 'min' or 'max' for metric
            filename_template: Template for checkpoint filenames
            best_filename: Filename for best checkpoint
            last_filename: Filename for last checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.filename_template = filename_template
        self.best_filename = best_filename
        self.last_filename = last_filename
        
        # Best metric tracking
        if mode == 'min':
            self.best_metric = float('inf')
            self.is_better = lambda x, best: x < best
        else:
            self.best_metric = float('-inf')
            self.is_better = lambda x, best: x > best
        
        # Checkpoint history
        self.checkpoints: List[str] = []
        self._load_existing_checkpoints()
        
        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoint files."""
        pattern = self.checkpoint_dir / 'checkpoint_epoch_*.pt'
        existing = sorted(glob.glob(str(pattern)))
        self.checkpoints = [os.path.basename(p) for p in existing]
        
        # Load best metric if best checkpoint exists
        best_path = self.checkpoint_dir / self.best_filename
        if best_path.exists():
            try:
                checkpoint = torch.load(best_path, map_location='cpu')
                if 'metrics' in checkpoint and self.metric_name in checkpoint['metrics']:
                    self.best_metric = checkpoint['metrics'][self.metric_name]
                    logger.info(f"Loaded best metric: {self.metric_name}={self.best_metric:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load best checkpoint: {e}")
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        epoch: int = 0,
        step: int = 0,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """
        Save checkpoint and update tracking.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: LR scheduler
            epoch: Current epoch
            step: Current step
            metrics: Metrics dictionary
            config: Model configuration
            **kwargs: Additional data
        
        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        
        # Generate filename
        filename = self.filename_template.format(epoch=epoch, step=step)
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=config,
            filepath=str(filepath),
            **kwargs,
        )
        
        # Update checkpoint list
        self.checkpoints.append(filename)
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / self.last_filename
        shutil.copy(filepath, last_path)
        
        # Check for best model
        if self.save_best and self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            
            if self.is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
                best_path = self.checkpoint_dir / self.best_filename
                shutil.copy(filepath, best_path)
                logger.info(f"New best model: {self.metric_name}={current_metric:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return str(filepath)
    
    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            oldest_path = self.checkpoint_dir / oldest
            
            if oldest_path.exists():
                oldest_path.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")
    
    def load(
        self,
        filename: str,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        device: str = 'cuda',
    ) -> Dict[str, Any]:
        """
        Load specific checkpoint.
        
        Args:
            filename: Checkpoint filename
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Target device
        
        Returns:
            Checkpoint dictionary
        """
        filepath = self.checkpoint_dir / filename
        return load_checkpoint(filepath, model, optimizer, scheduler, device)
    
    def load_best(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        device: str = 'cuda',
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        return self.load(self.best_filename, model, optimizer, scheduler, device)
    
    def load_last(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        device: str = 'cuda',
    ) -> Dict[str, Any]:
        """Load last checkpoint."""
        return self.load(self.last_filename, model, optimizer, scheduler, device)
    
    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        device: str = 'cuda',
    ) -> Tuple[int, Dict[str, float]]:
        """
        Resume training from last checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Target device
        
        Returns:
            (epoch, metrics) tuple
        """
        last_path = self.checkpoint_dir / self.last_filename
        
        if not last_path.exists():
            logger.warning("No checkpoint to resume from")
            return 0, {}
        
        checkpoint = self.load_last(model, optimizer, scheduler, device)
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Resumed from epoch {epoch}")
        
        return epoch, metrics
    
    def get_best_metric(self) -> float:
        """Get best metric value."""
        return self.best_metric
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoints."""
        return self.checkpoints.copy()
    
    def delete_checkpoint(self, filename: str):
        """Delete specific checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            if filename in self.checkpoints:
                self.checkpoints.remove(filename)
            logger.info(f"Deleted checkpoint: {filename}")


# =============================================================================
# Model Export
# =============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: Dict[str, Tuple],
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 14,
    device: str = 'cuda',
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_shapes: Dictionary of input shapes
        input_names: Names for inputs
        output_names: Names for outputs
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version
        device: Export device
    
    Returns:
        Path to ONNX file
    """
    model.eval()
    model.to(device)
    
    # Create dummy inputs
    dummy_inputs = []
    for name, shape in input_shapes.items():
        dummy_inputs.append(torch.randn(*shape, device=device))
    
    if len(dummy_inputs) == 1:
        dummy_input = dummy_inputs[0]
    else:
        dummy_input = tuple(dummy_inputs)
    
    # Default names
    input_names = input_names or list(input_shapes.keys())
    output_names = output_names or ['output']
    
    # Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    
    logger.info(f"ONNX model exported: {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")
    
    return output_path


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    method: str = 'trace',
    device: str = 'cuda',
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        example_inputs: Example inputs for tracing
        method: 'trace' or 'script'
        device: Export device
    
    Returns:
        Path to TorchScript file
    """
    model.eval()
    model.to(device)
    
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = example_inputs.to(device)
    else:
        example_inputs = tuple(x.to(device) for x in example_inputs)
    
    # Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with torch.no_grad():
        if method == 'trace':
            scripted = torch.jit.trace(model, example_inputs)
        else:
            scripted = torch.jit.script(model)
    
    scripted.save(output_path)
    
    logger.info(f"TorchScript model exported: {output_path}")
    
    return output_path


# =============================================================================
# Checkpoint Averaging (SWA)
# =============================================================================

def average_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    device: str = 'cpu',
) -> str:
    """
    Average multiple checkpoints (Stochastic Weight Averaging).
    
    Args:
        checkpoint_paths: List of checkpoint paths
        output_path: Output path for averaged checkpoint
        device: Device for loading
    
    Returns:
        Path to averaged checkpoint
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints to average")
    
    # Load first checkpoint as base
    avg_state = torch.load(checkpoint_paths[0], map_location=device)
    
    if 'model_state_dict' in avg_state:
        avg_state_dict = avg_state['model_state_dict']
    else:
        avg_state_dict = avg_state
    
    # Convert to float for averaging
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key].float()
    
    # Add remaining checkpoints
    for i, path in enumerate(checkpoint_paths[1:], 1):
        checkpoint = torch.load(path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key].float()
    
    # Average
    n = len(checkpoint_paths)
    for key in avg_state_dict:
        avg_state_dict[key] /= n
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    output = {
        'model_state_dict': avg_state_dict,
        'averaged_from': checkpoint_paths,
        'num_checkpoints': n,
        'timestamp': datetime.now().isoformat(),
    }
    
    torch.save(output, output_path)
    
    logger.info(f"Averaged {n} checkpoints: {output_path}")
    
    return output_path


def get_checkpoint_metric(checkpoint_path: str, metric_name: str) -> Optional[float]:
    """
    Get metric value from checkpoint.
    
    Args:
        checkpoint_path: Checkpoint path
        metric_name: Metric name
    
    Returns:
        Metric value or None
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metrics = checkpoint.get('metrics', {})
        return metrics.get(metric_name)
    except Exception:
        return None


def select_best_checkpoints(
    checkpoint_dir: str,
    metric_name: str,
    n: int = 5,
    mode: str = 'min',
) -> List[str]:
    """
    Select N best checkpoints by metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Metric to sort by
        n: Number of checkpoints to select
        mode: 'min' or 'max'
    
    Returns:
        List of checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
    
    # Get metrics
    checkpoint_metrics = []
    for path in checkpoints:
        metric = get_checkpoint_metric(str(path), metric_name)
        if metric is not None:
            checkpoint_metrics.append((str(path), metric))
    
    # Sort
    reverse = (mode == 'max')
    checkpoint_metrics.sort(key=lambda x: x[1], reverse=reverse)
    
    # Select top N
    selected = [path for path, _ in checkpoint_metrics[:n]]
    
    logger.info(f"Selected {len(selected)} best checkpoints by {metric_name}")
    
    return selected


# =============================================================================
# Model Loading Utilities
# =============================================================================

def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    exclude_patterns: List[str] = None,
    include_patterns: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load pretrained weights with flexible matching.
    
    Args:
        model: Target model
        checkpoint_path: Checkpoint path
        strict: Strict loading
        exclude_patterns: Regex patterns to exclude
        include_patterns: Regex patterns to include
    
    Returns:
        (matched_keys, missing_keys) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle module prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Filter keys
    if exclude_patterns:
        exclude_re = [re.compile(p) for p in exclude_patterns]
        state_dict = {
            k: v for k, v in state_dict.items()
            if not any(r.search(k) for r in exclude_re)
        }
    
    if include_patterns:
        include_re = [re.compile(p) for p in include_patterns]
        state_dict = {
            k: v for k, v in state_dict.items()
            if any(r.search(k) for r in include_re)
        }
    
    # Load
    model_state = model.state_dict()
    
    matched_keys = []
    missing_keys = []
    
    for key in state_dict:
        if key in model_state:
            if state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                matched_keys.append(key)
            else:
                missing_keys.append(key)
                logger.warning(f"Shape mismatch: {key}")
        else:
            missing_keys.append(key)
    
    model.load_state_dict(model_state, strict=False)
    
    logger.info(f"Loaded {len(matched_keys)} keys, {len(missing_keys)} missing")
    
    return matched_keys, missing_keys


def freeze_layers(
    model: nn.Module,
    freeze_patterns: List[str] = None,
    unfreeze_patterns: List[str] = None,
) -> int:
    """
    Freeze/unfreeze model layers by pattern.
    
    Args:
        model: PyTorch model
        freeze_patterns: Patterns to freeze
        unfreeze_patterns: Patterns to unfreeze
    
    Returns:
        Number of frozen parameters
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        if freeze_patterns:
            freeze_re = [re.compile(p) for p in freeze_patterns]
            should_freeze = any(r.search(name) for r in freeze_re)
        
        if unfreeze_patterns:
            unfreeze_re = [re.compile(p) for p in unfreeze_patterns]
            if any(r.search(name) for r in unfreeze_re):
                should_freeze = False
        
        param.requires_grad = not should_freeze
        
        if should_freeze:
            frozen_count += param.numel()
    
    logger.info(f"Frozen {frozen_count:,} parameters")
    
    return frozen_count


# =============================================================================
# Checkpoint Comparison
# =============================================================================

def compare_checkpoints(
    checkpoint1_path: str,
    checkpoint2_path: str,
) -> Dict[str, Any]:
    """
    Compare two checkpoints.
    
    Args:
        checkpoint1_path: First checkpoint path
        checkpoint2_path: Second checkpoint path
    
    Returns:
        Comparison results
    """
    cp1 = torch.load(checkpoint1_path, map_location='cpu')
    cp2 = torch.load(checkpoint2_path, map_location='cpu')
    
    state1 = cp1.get('model_state_dict', cp1)
    state2 = cp2.get('model_state_dict', cp2)
    
    # Key comparison
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    # Weight comparison
    weight_diffs = {}
    for key in common_keys:
        if state1[key].shape == state2[key].shape:
            diff = (state1[key] - state2[key]).abs()
            weight_diffs[key] = {
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'shape': list(state1[key].shape),
            }
    
    # Metric comparison
    metrics1 = cp1.get('metrics', {})
    metrics2 = cp2.get('metrics', {})
    
    return {
        'common_keys': len(common_keys),
        'only_in_checkpoint1': list(only_in_1),
        'only_in_checkpoint2': list(only_in_2),
        'weight_differences': weight_diffs,
        'metrics_checkpoint1': metrics1,
        'metrics_checkpoint2': metrics2,
        'epoch_checkpoint1': cp1.get('epoch'),
        'epoch_checkpoint2': cp2.get('epoch'),
    }
