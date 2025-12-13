"""
Learning rate schedulers for trajectory prediction training.

Supports various scheduling strategies:
- Step decay (Social-LSTM)
- Cosine annealing (transformers)
- Warmup + decay (transformer training)
- Reduce on plateau (adaptive)

References:
- Social-LSTM: Step decay every 20 epochs, gamma=0.5
- Transformers: Warmup + cosine decay
"""

import math
import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR,
)
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler.
    
    Linearly increases learning rate from 0 to base_lr over warmup_steps.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup followed by cosine annealing.
    
    Common for transformer training:
    1. Linear warmup from 0 to base_lr
    2. Cosine decay from base_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-8,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate after decay
            warmup_start_lr: Starting LR for warmup
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Warmup followed by linear decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                max(self.min_lr, base_lr * (1 - progress))
                for base_lr in self.base_lrs
            ]


class CurriculumScheduler(_LRScheduler):
    """
    Scheduler for curriculum learning.
    
    Adjusts learning rate based on difficulty/prediction horizon.
    Used for long-term (30s) trajectory prediction.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        lr_factors: List[float],
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            milestones: Epoch milestones for curriculum stages
            lr_factors: LR multipliers at each stage
            last_epoch: Last epoch index
        """
        self.milestones = milestones
        self.lr_factors = lr_factors
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Find current stage
        stage = 0
        for i, milestone in enumerate(self.milestones):
            if self.last_epoch >= milestone:
                stage = i + 1
        
        factor = self.lr_factors[min(stage, len(self.lr_factors) - 1)]
        return [base_lr * factor for base_lr in self.base_lrs]


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = 'cosine',
    total_epochs: int = 200,
    warmup_epochs: int = 10,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs (for warmup schedulers)
        **kwargs: Additional scheduler arguments
    
    Returns:
        Configured scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 50),
            gamma=kwargs.get('gamma', 0.5)
        )
    
    elif scheduler_name == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [100, 150, 180]),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_name == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.99)
        )
    
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_name == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=kwargs.get('min_lr', 1e-6),
            warmup_start_lr=kwargs.get('warmup_start_lr', 1e-8)
        )
    
    elif scheduler_name == 'warmup_linear':
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=kwargs.get('min_lr', 0.0)
        )
    
    elif scheduler_name == 'warmup':
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=warmup_epochs
        )
    
    elif scheduler_name == 'cosine_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 50),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6),
            verbose=kwargs.get('verbose', True)
        )
    
    elif scheduler_name == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            total_steps=kwargs.get('total_steps', total_epochs * 100),
            pct_start=kwargs.get('pct_start', 0.1),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    
    elif scheduler_name == 'none' or scheduler_name is None:
        # Constant LR (no scheduling)
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    logger.info(f"Created {scheduler_name} scheduler")
    
    return scheduler


def get_scheduler_from_config(
    optimizer: Optimizer,
    config: Dict
) -> _LRScheduler:
    """
    Create scheduler from configuration dictionary.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary with 'scheduler' section
    
    Returns:
        Configured scheduler
    
    Example config:
        scheduler:
            name: warmup_cosine
            warmup_epochs: 10
            min_lr: 1e-6
        training:
            epochs: 300
    """
    sched_config = config.get('scheduler', {})
    train_config = config.get('training', {})
    
    scheduler_name = sched_config.get('name', 'cosine')
    total_epochs = train_config.get('epochs', 200)
    warmup_epochs = sched_config.get('warmup_epochs', 10)
    
    # Collect kwargs
    kwargs = {}
    for key in ['step_size', 'gamma', 'milestones', 'min_lr', 'warmup_start_lr',
                'T_0', 'T_mult', 'patience', 'factor', 'mode', 'max_lr',
                'pct_start', 'anneal_strategy']:
        if key in sched_config:
            kwargs[key] = sched_config[key]
    
    return get_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        **kwargs
    )


# =============================================================================
# Preset Configurations
# =============================================================================

SCHEDULER_PRESETS = {
    # Social-LSTM style
    'social_lstm': {
        'name': 'step',
        'step_size': 20,
        'gamma': 0.5,
    },
    # Social-GAN style
    'social_gan': {
        'name': 'step',
        'step_size': 100,
        'gamma': 0.5,
    },
    # Transformer style
    'transformer': {
        'name': 'warmup_cosine',
        'warmup_epochs': 10,
        'min_lr': 1e-6,
    },
    # Diffusion style
    'diffusion': {
        'name': 'warmup_cosine',
        'warmup_epochs': 20,
        'min_lr': 1e-6,
    },
    # Long-term prediction
    'long_term': {
        'name': 'warmup_cosine',
        'warmup_epochs': 20,
        'min_lr': 1e-7,
    },
}


def get_preset_scheduler(
    optimizer: Optimizer,
    preset: str = 'transformer',
    total_epochs: int = 200
) -> _LRScheduler:
    """
    Create scheduler from preset configuration.
    
    Args:
        optimizer: Optimizer to schedule
        preset: Preset name
        total_epochs: Total training epochs
    
    Returns:
        Configured scheduler
    """
    if preset not in SCHEDULER_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available: {list(SCHEDULER_PRESETS.keys())}")
    
    config = SCHEDULER_PRESETS[preset].copy()
    name = config.pop('name')
    warmup_epochs = config.pop('warmup_epochs', 0)
    
    logger.info(f"Using '{preset}' scheduler preset")
    
    return get_scheduler(
        optimizer,
        scheduler_name=name,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        **config
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_current_lr(optimizer: Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_lr(optimizer: Optimizer, lr: float):
    """
    Set learning rate for all parameter groups.
    
    Args:
        optimizer: Optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_on_loss(
    optimizer: Optimizer,
    current_loss: float,
    best_loss: float,
    factor: float = 0.5,
    threshold: float = 0.01
) -> Tuple[float, bool]:
    """
    Manually adjust LR if loss hasn't improved.
    
    Args:
        optimizer: Optimizer
        current_loss: Current epoch loss
        best_loss: Best loss so far
        factor: LR reduction factor
        threshold: Improvement threshold
    
    Returns:
        (new_best_loss, was_lr_reduced)
    """
    if current_loss < best_loss - threshold:
        return current_loss, False
    
    current_lr = get_current_lr(optimizer)
    new_lr = current_lr * factor
    set_lr(optimizer, new_lr)
    
    logger.info(f"Reduced LR from {current_lr:.6f} to {new_lr:.6f}")
    
    return best_loss, True


class SchedulerWrapper:
    """
    Wrapper to handle both epoch-based and step-based schedulers.
    """
    
    def __init__(
        self,
        scheduler: _LRScheduler,
        step_per_batch: bool = False
    ):
        """
        Args:
            scheduler: Base scheduler
            step_per_batch: If True, step scheduler every batch instead of epoch
        """
        self.scheduler = scheduler
        self.step_per_batch = step_per_batch
    
    def step_batch(self):
        """Call after each training batch."""
        if self.step_per_batch:
            self.scheduler.step()
    
    def step_epoch(self, val_loss: Optional[float] = None):
        """
        Call after each epoch.
        
        Args:
            val_loss: Validation loss (for ReduceLROnPlateau)
        """
        if not self.step_per_batch:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()[0]
        return get_current_lr(self.scheduler.optimizer)
    
    def state_dict(self):
        """Get scheduler state."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)
