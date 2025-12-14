"""
Optimizer factory for trajectory prediction models.

Supports various optimizers with common configurations:
- Adam/AdamW (default for transformers)
- SGD with momentum
- RAdam (rectified Adam)
- LAMB (Layer-wise Adaptive Moments)

References:
- Social-LSTM uses SGD with lr=0.003
- Social-GAN uses Adam with lr=0.001
- Transformer models typically use AdamW with lr=7.5e-4
"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Union, Iterator
import logging

logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd', 'radam')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Configured optimizer
    """
    # Get parameters (filter out those that don't require grad)
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
        )
    
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
        )
    
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True),
        )
    
    elif optimizer_name == 'radam':
        optimizer = optim.RAdam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
        )
    
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.0),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Created {optimizer_name.upper()} optimizer with lr={lr}, "
                f"weight_decay={weight_decay}")
    
    return optimizer


def get_optimizer_with_param_groups(
    model: torch.nn.Module,
    optimizer_name: str = 'adamw',
    base_lr: float = 1e-3,
    encoder_lr_factor: float = 0.1,
    weight_decay: float = 0.01,
    no_decay_params: List[str] = None,
    **kwargs
) -> Optimizer:
    """
    Create optimizer with different learning rates for different parameter groups.
    
    Useful for fine-tuning pretrained models where you want lower LR for encoder.
    
    Args:
        model: PyTorch model with 'encoder' and other components
        optimizer_name: Name of optimizer
        base_lr: Base learning rate
        encoder_lr_factor: Factor to multiply base_lr for encoder params
        weight_decay: Weight decay
        no_decay_params: Parameter names to exclude from weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Configured optimizer with parameter groups
    """
    no_decay_params = no_decay_params or ['bias', 'LayerNorm', 'layer_norm']
    
    # Group parameters
    encoder_params_decay = []
    encoder_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if encoder parameter
        is_encoder = 'encoder' in name.lower()
        
        # Check if should have weight decay
        has_decay = not any(nd in name for nd in no_decay_params)
        
        if is_encoder:
            if has_decay:
                encoder_params_decay.append(param)
            else:
                encoder_params_no_decay.append(param)
        else:
            if has_decay:
                other_params_decay.append(param)
            else:
                other_params_no_decay.append(param)
    
    # Create parameter groups
    param_groups = [
        {
            'params': encoder_params_decay,
            'lr': base_lr * encoder_lr_factor,
            'weight_decay': weight_decay,
            'name': 'encoder_decay'
        },
        {
            'params': encoder_params_no_decay,
            'lr': base_lr * encoder_lr_factor,
            'weight_decay': 0.0,
            'name': 'encoder_no_decay'
        },
        {
            'params': other_params_decay,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other_decay'
        },
        {
            'params': other_params_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'other_no_decay'
        },
    ]
    
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # Log parameter counts
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        logger.info(f"Param group '{group['name']}': {num_params:,} params, "
                    f"lr={group['lr']}, wd={group['weight_decay']}")
    
    # Create optimizer
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def get_optimizer_from_config(
    model: torch.nn.Module,
    config: Dict
) -> Optimizer:
    """
    Create optimizer from configuration dictionary.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary with 'optimizer' section
    
    Returns:
        Configured optimizer
    
    Example config:
        optimizer:
            name: adamw
            lr: 0.00075
            weight_decay: 0.01
            betas: [0.9, 0.999]
    """
    opt_config = config.get('optimizer', {})
    
    optimizer_name = opt_config.get('name', 'adam')
    lr = opt_config.get('lr', 1e-3)
    weight_decay = opt_config.get('weight_decay', 0.0)
    
    # Extract additional kwargs
    kwargs = {}
    if 'betas' in opt_config:
        kwargs['betas'] = tuple(opt_config['betas'])
    if 'eps' in opt_config:
        kwargs['eps'] = opt_config['eps']
    if 'momentum' in opt_config:
        kwargs['momentum'] = opt_config['momentum']
    
    # Check for parameter groups
    if opt_config.get('use_param_groups', False):
        return get_optimizer_with_param_groups(
            model,
            optimizer_name=optimizer_name,
            base_lr=lr,
            encoder_lr_factor=opt_config.get('encoder_lr_factor', 0.1),
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        return get_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )


class GradientClipper:
    """
    Gradient clipping utility.
    
    Supports both norm-based and value-based clipping.
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        clip_value: Optional[float] = None,
        norm_type: float = 2.0
    ):
        """
        Args:
            max_norm: Maximum gradient norm for norm clipping
            clip_value: Maximum gradient value for value clipping
            norm_type: Norm type for norm clipping
        """
        self.max_norm = max_norm
        self.clip_value = clip_value
        self.norm_type = norm_type
    
    def __call__(
        self,
        parameters: Iterator[torch.nn.Parameter]
    ) -> Optional[float]:
        """
        Clip gradients.
        
        Args:
            parameters: Model parameters iterator
        
        Returns:
            Gradient norm before clipping (if norm clipping)
        """
        params = list(filter(lambda p: p.grad is not None, parameters))
        
        if len(params) == 0:
            return None
        
        if self.clip_value is not None:
            for param in params:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)
            return None
        
        # Norm clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            params, self.max_norm, norm_type=self.norm_type
        )
        
        return total_norm.item()


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model parameters for stable evaluation.
    Used in diffusion models and other generative approaches.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
            device: Device for shadow parameters
        """
        self.decay = decay
        self.device = device
        
        # Create shadow parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self, model: torch.nn.Module):
        """
        Update shadow parameters with current model parameters.
        
        Args:
            model: Model with updated parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data
                    )
    
    def apply(self, model: torch.nn.Module):
        """
        Apply shadow parameters to model.
        
        Args:
            model: Model to update
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])
    
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]):
        """
        Restore model parameters from backup.
        
        Args:
            model: Model to restore
            backup: Backup of original parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    
    def store(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Store current model parameters.
        
        Args:
            model: Model to store
        
        Returns:
            Dictionary of parameter backups
        """
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
        return backup


# =============================================================================
# Preset Configurations
# =============================================================================

OPTIMIZER_PRESETS = {
    # Social-LSTM style
    'social_lstm': {
        'name': 'sgd',
        'lr': 0.003,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    },
    # Social-GAN style
    'social_gan': {
        'name': 'adam',
        'lr': 0.001,
        'betas': (0.5, 0.999),
        'weight_decay': 0.0,
    },
    # Transformer style (Social-Pose, Human Scene Transformer)
    'transformer': {
        'name': 'adamw',
        'lr': 0.00075,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
    },
    # Diffusion style
    'diffusion': {
        'name': 'adamw',
        'lr': 0.0001,
        'betas': (0.9, 0.999),
        'weight_decay': 0.0,
    },
}


def get_preset_optimizer(
    model: torch.nn.Module,
    preset: str = 'transformer'
) -> Optimizer:
    """
    Create optimizer from preset configuration.
    
    Args:
        model: PyTorch model
        preset: Preset name ('social_lstm', 'social_gan', 'transformer', 'diffusion')
    
    Returns:
        Configured optimizer
    """
    if preset not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. "
                        f"Available: {list(OPTIMIZER_PRESETS.keys())}")
    
    config = OPTIMIZER_PRESETS[preset].copy()
    name = config.pop('name')
    lr = config.pop('lr')
    weight_decay = config.pop('weight_decay', 0.0)
    
    logger.info(f"Using '{preset}' optimizer preset")
    
    return get_optimizer(
        model,
        optimizer_name=name,
        lr=lr,
        weight_decay=weight_decay,
        **config
    )
