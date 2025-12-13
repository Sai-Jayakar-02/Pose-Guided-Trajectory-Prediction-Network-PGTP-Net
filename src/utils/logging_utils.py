"""
Logging Utilities for Training and Evaluation.

Provides unified interface for multiple logging backends:
- TensorBoard: Local visualization
- Weights & Biases (W&B): Cloud-based experiment tracking
- Console: Rich console output
- File: JSON/CSV logging

Usage:
    from src.utils import Logger, TensorBoardLogger, WandBLogger
    
    # TensorBoard
    logger = TensorBoardLogger(log_dir='runs/exp1')
    logger.log_scalar('loss', 0.5, step=100)
    logger.log_histogram('weights', model.fc.weight, step=100)
    
    # Weights & Biases
    logger = WandBLogger(project='trajectory-prediction', name='social-pose-v1')
    logger.log({'loss': 0.5, 'ade': 0.41}, step=100)
    
    # Unified logger (combines multiple backends)
    logger = Logger(
        backends=['tensorboard', 'wandb', 'console'],
        log_dir='runs/exp1',
        project='trajectory-prediction',
    )
    logger.log_metrics({'loss': 0.5}, step=100)

References:
- TensorBoard: https://www.tensorflow.org/tensorboard
- Weights & Biases: https://wandb.ai/
"""

import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Base Logger
# =============================================================================

class BaseLogger:
    """Base class for logging backends."""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._step = 0
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log a scalar value."""
        raise NotImplementedError
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int = None):
        """Log multiple scalars under a tag."""
        raise NotImplementedError
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log a histogram."""
        raise NotImplementedError
    
    def log_image(self, tag: str, image: np.ndarray, step: int = None):
        """Log an image."""
        raise NotImplementedError
    
    def log_figure(self, tag: str, figure, step: int = None):
        """Log a matplotlib figure."""
        raise NotImplementedError
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Log text."""
        raise NotImplementedError
    
    def log_hyperparams(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        """Log hyperparameters."""
        raise NotImplementedError
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step)
    
    def set_step(self, step: int):
        """Set current step."""
        self._step = step
    
    def close(self):
        """Close logger and flush buffers."""
        pass


# =============================================================================
# TensorBoard Logger
# =============================================================================

class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logging backend.
    
    Features:
    - Scalars, histograms, images, text
    - Hyperparameter logging
    - Custom layouts
    - Embedding projector
    """
    
    def __init__(
        self,
        log_dir: str = 'runs',
        comment: str = '',
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Log directory
            comment: Comment to append to folder name
            purge_step: Purge events after this step
            max_queue: Max queue size before flushing
            flush_secs: Flush interval in seconds
        """
        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if comment:
            run_name = f'{timestamp}_{comment}'
        else:
            run_name = timestamp
        
        super().__init__(os.path.join(log_dir, run_name))
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
            )
            logger.info(f"TensorBoard logging to {self.log_dir}")
            
        except ImportError:
            raise ImportError(
                "TensorBoard not installed. Install with: pip install tensorboard"
            )
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar value."""
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int = None):
        """Log multiple scalars under same tag."""
        step = step if step is not None else self._step
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log histogram of values."""
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int = None):
        """
        Log image.
        
        Args:
            tag: Image tag
            image: Image array (H, W, C) or (C, H, W)
            step: Global step
        """
        step = step if step is not None else self._step
        
        # Handle different formats
        if image.ndim == 3:
            if image.shape[0] in [1, 3, 4]:  # (C, H, W)
                pass
            else:  # (H, W, C)
                image = np.transpose(image, (2, 0, 1))
        
        self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: np.ndarray, step: int = None):
        """Log multiple images as grid."""
        step = step if step is not None else self._step
        self.writer.add_images(tag, images, step)
    
    def log_figure(self, tag: str, figure, step: int = None):
        """Log matplotlib figure."""
        step = step if step is not None else self._step
        self.writer.add_figure(tag, figure, step)
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Log text."""
        step = step if step is not None else self._step
        self.writer.add_text(tag, text, step)
    
    def log_hyperparams(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        """Log hyperparameters with optional metrics."""
        self.writer.add_hparams(params, metrics or {})
    
    def log_embedding(
        self,
        tag: str,
        embeddings: np.ndarray,
        metadata: List[str] = None,
        label_img: np.ndarray = None,
        step: int = None,
    ):
        """Log embeddings for projector."""
        step = step if step is not None else self._step
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            label_img=label_img,
            global_step=step,
            tag=tag,
        )
    
    def log_graph(self, model, input_to_model=None):
        """Log model graph."""
        self.writer.add_graph(model, input_to_model)
    
    def log_pr_curve(
        self,
        tag: str,
        labels: np.ndarray,
        predictions: np.ndarray,
        step: int = None,
    ):
        """Log precision-recall curve."""
        step = step if step is not None else self._step
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def flush(self):
        """Flush writer buffer."""
        self.writer.flush()
    
    def close(self):
        """Close writer."""
        self.writer.close()


# =============================================================================
# Weights & Biases Logger
# =============================================================================

class WandBLogger(BaseLogger):
    """
    Weights & Biases logging backend.
    
    Features:
    - Cloud-based experiment tracking
    - Rich visualizations
    - Model versioning
    - Hyperparameter sweeps
    - Team collaboration
    """
    
    def __init__(
        self,
        project: str = 'trajectory-prediction',
        name: str = None,
        config: Dict[str, Any] = None,
        tags: List[str] = None,
        notes: str = None,
        group: str = None,
        job_type: str = None,
        resume: str = None,
        mode: str = 'online',
        log_dir: str = None,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes
            group: Group name for related runs
            job_type: Job type (train, eval, etc.)
            resume: Resume mode ('allow', 'must', 'never', run_id)
            mode: 'online', 'offline', or 'disabled'
            log_dir: Local log directory
        """
        super().__init__(log_dir)
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )
        
        # Initialize run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            job_type=job_type,
            resume=resume,
            mode=mode,
            dir=log_dir,
        )
        
        logger.info(f"W&B run initialized: {self.run.name}")
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar value."""
        step = step if step is not None else self._step
        self.wandb.log({tag: value}, step=step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int = None):
        """Log multiple scalars."""
        step = step if step is not None else self._step
        prefixed = {f'{tag}/{k}': v for k, v in values.items()}
        self.wandb.log(prefixed, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log histogram."""
        step = step if step is not None else self._step
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int = None, caption: str = None):
        """Log image."""
        step = step if step is not None else self._step
        self.wandb.log({tag: self.wandb.Image(image, caption=caption)}, step=step)
    
    def log_images(self, tag: str, images: List[np.ndarray], step: int = None, captions: List[str] = None):
        """Log multiple images."""
        step = step if step is not None else self._step
        wandb_images = [
            self.wandb.Image(img, caption=cap if captions else None)
            for img, cap in zip(images, captions or [None] * len(images))
        ]
        self.wandb.log({tag: wandb_images}, step=step)
    
    def log_figure(self, tag: str, figure, step: int = None):
        """Log matplotlib figure."""
        step = step if step is not None else self._step
        self.wandb.log({tag: self.wandb.Image(figure)}, step=step)
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Log text."""
        step = step if step is not None else self._step
        self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        """Update config with hyperparameters."""
        self.wandb.config.update(params)
        if metrics:
            self.wandb.summary.update(metrics)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        step = step if step is not None else self._step
        self.wandb.log(metrics, step=step)
    
    def log_table(self, tag: str, columns: List[str], data: List[List[Any]], step: int = None):
        """Log table."""
        step = step if step is not None else self._step
        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({tag: table}, step=step)
    
    def log_video(self, tag: str, video: np.ndarray, fps: int = 4, step: int = None):
        """Log video."""
        step = step if step is not None else self._step
        self.wandb.log({tag: self.wandb.Video(video, fps=fps)}, step=step)
    
    def log_artifact(self, name: str, artifact_type: str, path: str):
        """Log artifact (model, dataset, etc.)."""
        artifact = self.wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def log_model(self, name: str, model_path: str):
        """Log model checkpoint."""
        self.log_artifact(name, 'model', model_path)
    
    def watch(self, model, log: str = 'all', log_freq: int = 100):
        """Watch model gradients and parameters."""
        self.wandb.watch(model, log=log, log_freq=log_freq)
    
    def alert(self, title: str, text: str, level: str = 'INFO'):
        """Send alert."""
        self.wandb.alert(title=title, text=text, level=level)
    
    def finish(self):
        """Finish run."""
        self.wandb.finish()
    
    def close(self):
        """Close logger."""
        self.finish()


# =============================================================================
# Console Logger
# =============================================================================

class ConsoleLogger(BaseLogger):
    """
    Console logging with rich formatting.
    
    Provides formatted console output with:
    - Progress bars
    - Tables
    - Colored output
    """
    
    def __init__(
        self,
        log_dir: str = None,
        print_freq: int = 1,
        use_rich: bool = True,
    ):
        """
        Initialize console logger.
        
        Args:
            log_dir: Directory for log files
            print_freq: Print frequency (log every N steps)
            use_rich: Use rich library for formatting
        """
        super().__init__(log_dir)
        self.print_freq = print_freq
        self.use_rich = use_rich
        
        self._metrics_buffer = defaultdict(list)
        self._last_print_step = 0
        
        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table
                self.console = Console()
                self.Table = Table
            except ImportError:
                self.use_rich = False
                self.console = None
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar to buffer."""
        step = step if step is not None else self._step
        self._metrics_buffer[tag].append(value)
        
        if step - self._last_print_step >= self.print_freq:
            self._flush_buffer(step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics and print."""
        step = step if step is not None else self._step
        
        for key, value in metrics.items():
            self._metrics_buffer[key].append(value)
        
        self._flush_buffer(step)
    
    def _flush_buffer(self, step: int):
        """Flush metrics buffer to console."""
        if not self._metrics_buffer:
            return
        
        # Compute averages
        avg_metrics = {
            k: np.mean(v) for k, v in self._metrics_buffer.items()
        }
        
        # Format output
        if self.use_rich and self.console:
            table = self.Table(title=f"Step {step}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in sorted(avg_metrics.items()):
                table.add_row(key, f"{value:.4f}")
            
            self.console.print(table)
        else:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in sorted(avg_metrics.items())])
            print(f"[Step {step}] {metrics_str}")
        
        self._metrics_buffer.clear()
        self._last_print_step = step
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Print text to console."""
        if self.use_rich and self.console:
            self.console.print(f"[bold]{tag}:[/bold] {text}")
        else:
            print(f"{tag}: {text}")
    
    def log_hyperparams(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        """Print hyperparameters."""
        if self.use_rich and self.console:
            table = self.Table(title="Hyperparameters")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in sorted(params.items()):
                table.add_row(key, str(value))
            
            self.console.print(table)
        else:
            print("Hyperparameters:")
            for key, value in sorted(params.items()):
                print(f"  {key}: {value}")


# =============================================================================
# File Logger
# =============================================================================

class FileLogger(BaseLogger):
    """
    File-based logging (JSON/CSV).
    
    Logs metrics to files for later analysis.
    """
    
    def __init__(
        self,
        log_dir: str,
        filename: str = 'metrics',
        format: str = 'json',
    ):
        """
        Initialize file logger.
        
        Args:
            log_dir: Log directory
            filename: Base filename (without extension)
            format: 'json' or 'csv'
        """
        super().__init__(log_dir)
        self.format = format
        
        if format == 'json':
            self.filepath = self.log_dir / f'{filename}.json'
            self._data = []
        elif format == 'csv':
            self.filepath = self.log_dir / f'{filename}.csv'
            self._data = []
            self._columns = set()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar to file."""
        step = step if step is not None else self._step
        self._add_entry({tag: value, 'step': step})
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to file."""
        step = step if step is not None else self._step
        entry = {'step': step, 'timestamp': datetime.now().isoformat()}
        entry.update(metrics)
        self._add_entry(entry)
    
    def _add_entry(self, entry: Dict):
        """Add entry to data."""
        self._data.append(entry)
        
        if self.format == 'csv':
            self._columns.update(entry.keys())
    
    def flush(self):
        """Write data to file."""
        if self.format == 'json':
            with open(self.filepath, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
        
        elif self.format == 'csv':
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(self._columns))
                writer.writeheader()
                writer.writerows(self._data)
    
    def close(self):
        """Flush and close."""
        self.flush()


# =============================================================================
# Unified Logger
# =============================================================================

class Logger:
    """
    Unified logger combining multiple backends.
    
    Usage:
        logger = Logger(
            backends=['tensorboard', 'wandb', 'console'],
            log_dir='runs/exp1',
            project='trajectory-prediction',
        )
        
        logger.log_metrics({'loss': 0.5, 'ade': 0.41}, step=100)
        logger.log_hyperparams({'lr': 0.001, 'batch_size': 64})
    """
    
    BACKEND_CLASSES = {
        'tensorboard': TensorBoardLogger,
        'wandb': WandBLogger,
        'console': ConsoleLogger,
        'file': FileLogger,
    }
    
    def __init__(
        self,
        backends: List[str] = ['tensorboard', 'console'],
        log_dir: str = 'runs',
        **kwargs,
    ):
        """
        Initialize unified logger.
        
        Args:
            backends: List of backend names
            log_dir: Log directory
            **kwargs: Backend-specific arguments
        """
        self.backends = backends
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.loggers = []
        
        for backend in backends:
            if backend not in self.BACKEND_CLASSES:
                logger.warning(f"Unknown backend: {backend}")
                continue
            
            try:
                backend_class = self.BACKEND_CLASSES[backend]
                
                # Extract backend-specific kwargs
                backend_kwargs = {'log_dir': str(self.log_dir)}
                
                if backend == 'wandb':
                    backend_kwargs['project'] = kwargs.get('project', 'trajectory-prediction')
                    backend_kwargs['name'] = kwargs.get('name')
                    backend_kwargs['config'] = kwargs.get('config')
                    backend_kwargs['tags'] = kwargs.get('tags')
                
                elif backend == 'tensorboard':
                    backend_kwargs['comment'] = kwargs.get('comment', '')
                
                elif backend == 'console':
                    backend_kwargs['print_freq'] = kwargs.get('print_freq', 1)
                
                elif backend == 'file':
                    backend_kwargs['format'] = kwargs.get('file_format', 'json')
                
                self.loggers.append(backend_class(**backend_kwargs))
                
            except Exception as e:
                logger.warning(f"Failed to initialize {backend}: {e}")
        
        logger.info(f"Logger initialized with backends: {[type(l).__name__ for l in self.loggers]}")
    
    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log scalar to all backends."""
        for log in self.loggers:
            try:
                log.log_scalar(tag, value, step)
            except Exception as e:
                logger.debug(f"Failed to log scalar to {type(log).__name__}: {e}")
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int = None):
        """Log multiple scalars."""
        for log in self.loggers:
            try:
                log.log_scalars(tag, values, step)
            except Exception as e:
                logger.debug(f"Failed to log scalars to {type(log).__name__}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to all backends."""
        for log in self.loggers:
            try:
                log.log_metrics(metrics, step)
            except Exception as e:
                logger.debug(f"Failed to log metrics to {type(log).__name__}: {e}")
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log histogram."""
        for log in self.loggers:
            try:
                log.log_histogram(tag, values, step)
            except Exception as e:
                logger.debug(f"Failed to log histogram to {type(log).__name__}: {e}")
    
    def log_image(self, tag: str, image: np.ndarray, step: int = None):
        """Log image."""
        for log in self.loggers:
            try:
                log.log_image(tag, image, step)
            except Exception as e:
                logger.debug(f"Failed to log image to {type(log).__name__}: {e}")
    
    def log_figure(self, tag: str, figure, step: int = None):
        """Log matplotlib figure."""
        for log in self.loggers:
            try:
                log.log_figure(tag, figure, step)
            except Exception as e:
                logger.debug(f"Failed to log figure to {type(log).__name__}: {e}")
    
    def log_text(self, tag: str, text: str, step: int = None):
        """Log text."""
        for log in self.loggers:
            try:
                log.log_text(tag, text, step)
            except Exception as e:
                logger.debug(f"Failed to log text to {type(log).__name__}: {e}")
    
    def log_hyperparams(self, params: Dict[str, Any], metrics: Dict[str, float] = None):
        """Log hyperparameters."""
        for log in self.loggers:
            try:
                log.log_hyperparams(params, metrics)
            except Exception as e:
                logger.debug(f"Failed to log hyperparams to {type(log).__name__}: {e}")
    
    def set_step(self, step: int):
        """Set current step for all backends."""
        for log in self.loggers:
            log.set_step(step)
    
    def flush(self):
        """Flush all backends."""
        for log in self.loggers:
            if hasattr(log, 'flush'):
                log.flush()
    
    def close(self):
        """Close all backends."""
        for log in self.loggers:
            try:
                log.close()
            except Exception as e:
                logger.debug(f"Failed to close {type(log).__name__}: {e}")


# =============================================================================
# Utility Functions
# =============================================================================

def setup_logging(
    log_dir: str = 'runs',
    backends: List[str] = ['tensorboard', 'console'],
    **kwargs,
) -> Logger:
    """
    Convenience function to set up logging.
    
    Args:
        log_dir: Log directory
        backends: Logging backends
        **kwargs: Backend-specific arguments
    
    Returns:
        Configured Logger instance
    """
    return Logger(backends=backends, log_dir=log_dir, **kwargs)


def log_model_summary(logger: Logger, model, input_size: tuple, step: int = 0):
    """
    Log model summary to logger.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        input_size: Input tensor size
        step: Global step
    """
    try:
        from torchinfo import summary
        model_summary = str(summary(model, input_size=input_size, verbose=0))
        logger.log_text('model_summary', model_summary, step)
    except ImportError:
        # Fallback to basic summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary_text = f"Total params: {total_params:,}\nTrainable: {trainable_params:,}"
        logger.log_text('model_summary', summary_text, step)


class MetricTracker:
    """
    Track metrics over time with moving averages.
    
    Usage:
        tracker = MetricTracker(['loss', 'ade', 'fde'])
        tracker.update('loss', 0.5)
        tracker.update('ade', 0.41)
        
        print(tracker.avg('loss'))
        tracker.reset()
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics or []
        self._data = defaultdict(lambda: {'sum': 0, 'count': 0, 'values': []})
    
    def update(self, name: str, value: float, n: int = 1):
        """Update metric."""
        self._data[name]['sum'] += value * n
        self._data[name]['count'] += n
        self._data[name]['values'].append(value)
    
    def avg(self, name: str) -> float:
        """Get average of metric."""
        data = self._data[name]
        if data['count'] == 0:
            return 0.0
        return data['sum'] / data['count']
    
    def sum(self, name: str) -> float:
        """Get sum of metric."""
        return self._data[name]['sum']
    
    def count(self, name: str) -> int:
        """Get count of metric."""
        return self._data[name]['count']
    
    def values(self, name: str) -> List[float]:
        """Get all values of metric."""
        return self._data[name]['values']
    
    def get_averages(self) -> Dict[str, float]:
        """Get averages of all metrics."""
        return {name: self.avg(name) for name in self._data}
    
    def reset(self, name: str = None):
        """Reset metric(s)."""
        if name:
            self._data[name] = {'sum': 0, 'count': 0, 'values': []}
        else:
            self._data.clear()
    
    def __str__(self) -> str:
        """String representation."""
        parts = [f"{name}: {self.avg(name):.4f}" for name in self._data]
        return " | ".join(parts)
