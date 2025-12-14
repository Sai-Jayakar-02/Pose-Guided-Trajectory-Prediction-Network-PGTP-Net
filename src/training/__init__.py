"""
Training module for trajectory prediction.

Components:
- losses: ADE, FDE, variety loss, collision loss
- optimizers: Optimizer factory with presets
- schedulers: LR scheduler factory with presets
- trainer: Main training loop with validation and checkpointing

Usage:
    from src.training import Trainer, TrajectoryLoss, create_trainer_from_config
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    # Train
    history = trainer.train()
    
    # Or use config-based creation
    trainer = create_trainer_from_config(model, train_loader, val_loader, config)
    history = trainer.train()
"""

# Loss functions
from .losses import (
    ADELoss,
    FDELoss,
    BestOfKLoss,
    VarietyLoss,
    CollisionLoss,
    SpeedConsistencyLoss,
    GaussianNLLLoss,
    TrajectoryLoss,
    compute_ade,
    compute_fde,
    compute_collision_rate,
)

# Optimizers
from .optimizers import (
    get_optimizer,
    get_optimizer_with_param_groups,
    get_optimizer_from_config,
    get_preset_optimizer,
    GradientClipper,
    EMA,
    OPTIMIZER_PRESETS,
)

# Schedulers
from .schedulers import (
    WarmupScheduler,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    CurriculumScheduler,
    get_scheduler,
    get_scheduler_from_config,
    get_preset_scheduler,
    SchedulerWrapper,
    get_current_lr,
    set_lr,
    SCHEDULER_PRESETS,
)

# Trainer
from .trainer import (
    Trainer,
    create_trainer_from_config,
    train_model,
)

__all__ = [
    # Losses
    'ADELoss',
    'FDELoss',
    'BestOfKLoss',
    'VarietyLoss',
    'CollisionLoss',
    'SpeedConsistencyLoss',
    'GaussianNLLLoss',
    'TrajectoryLoss',
    'compute_ade',
    'compute_fde',
    'compute_collision_rate',
    
    # Optimizers
    'get_optimizer',
    'get_optimizer_with_param_groups',
    'get_optimizer_from_config',
    'get_preset_optimizer',
    'GradientClipper',
    'EMA',
    'OPTIMIZER_PRESETS',
    
    # Schedulers
    'WarmupScheduler',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler',
    'CurriculumScheduler',
    'get_scheduler',
    'get_scheduler_from_config',
    'get_preset_scheduler',
    'SchedulerWrapper',
    'get_current_lr',
    'set_lr',
    'SCHEDULER_PRESETS',
    
    # Trainer
    'Trainer',
    'create_trainer_from_config',
    'train_model',
]
