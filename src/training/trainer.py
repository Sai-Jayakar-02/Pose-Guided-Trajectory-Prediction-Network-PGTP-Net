"""
Trainer for trajectory prediction models.

Features:
- Training loop with validation
- Checkpointing (best model, periodic saves)
- Early stopping
- Mixed precision training (AMP)
- Gradient accumulation
- Logging (TensorBoard, WandB)
- Multi-GPU support (DataParallel)

References:
- Best practices from Social-LSTM, Social-GAN, Social-Pose papers
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable, Tuple, List, Any
from pathlib import Path
import logging
from tqdm import tqdm
from collections import defaultdict

from .losses import TrajectoryLoss, compute_ade, compute_fde, compute_collision_rate
from .optimizers import get_optimizer, get_optimizer_from_config, GradientClipper, EMA
from .schedulers import get_scheduler, get_scheduler_from_config, SchedulerWrapper, get_current_lr

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for trajectory prediction models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        device: torch.device = None,
        config: Dict = None,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function (default: TrajectoryLoss)
            device: Device to train on
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        
        # Device
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = loss_fn or TrajectoryLoss(
            ade_weight=config.get('loss', {}).get('ade_weight', 1.0),
            fde_weight=config.get('loss', {}).get('fde_weight', 1.0),
            variety_weight=config.get('loss', {}).get('variety_weight', 0.5),
            use_best_of_k=config.get('loss', {}).get('use_best_of_k', True),
            k=config.get('model', {}).get('k_samples', 20),
        )
        
        # Training settings
        train_config = self.config.get('training', {})
        self.epochs = train_config.get('epochs', 200)
        self.grad_clip = train_config.get('grad_clip', 1.0)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation', 1)
        self.use_amp = train_config.get('mixed_precision', False)
        self.log_interval = train_config.get('log_interval', 10)
        
        # Gradient clipping
        self.gradient_clipper = GradientClipper(max_norm=self.grad_clip) if self.grad_clip > 0 else None
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(train_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_config.get('save_every', 10)
        
        # Early stopping
        self.patience = train_config.get('early_stopping_patience', 20)
        self.min_delta = train_config.get('early_stopping_delta', 0.0)
        
        # EMA (optional)
        self.use_ema = train_config.get('use_ema', False)
        self.ema = EMA(model, decay=train_config.get('ema_decay', 0.9999)) if self.use_ema else None
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_ade = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = defaultdict(list)
        
        # Logging
        self.writer = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.config.get('training', {}).get('log_dir', 'logs')
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}, "
                   f"Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, SchedulerWrapper):
                    self.scheduler.step_epoch(val_metrics.get('loss'))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Check for improvement
            improved = self._check_improvement(val_metrics)
            
            # Save checkpoint
            if improved or (self.save_every > 0 and (epoch + 1) % self.save_every == 0):
                self._save_checkpoint(is_best=improved)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        logger.info(f"Best validation ADE: {self.best_val_ade:.4f}")
        
        # Close TensorBoard
        if self.writer is not None:
            self.writer.close()
        
        return dict(self.history)
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self._to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                loss_dict = self._forward_step(batch)
                loss = loss_dict['total'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clipper is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    self.gradient_clipper(self.model.parameters())
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # EMA update
                if self.ema is not None:
                    self.ema.update(self.model)
                
                self.global_step += 1
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'ade': f"{loss_dict.get('ade', 0):.4f}",
                    'lr': f"{get_current_lr(self.optimizer):.6f}"
                })
        
        # Average losses
        metrics = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return metrics
    
    def _validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Use EMA model for validation if available
        if self.ema is not None:
            backup = self.ema.store(self.model)
            self.ema.apply(self.model)
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._to_device(batch)
                
                # Forward pass
                loss_dict = self._forward_step(batch)
                
                # Compute additional metrics
                pred = self._get_predictions(batch)
                target = batch['pred_traj']
                
                ade = compute_ade(pred, target, best_of_k=True)
                fde = compute_fde(pred, target, best_of_k=True)
                
                epoch_metrics['loss'] += loss_dict['total'].item()
                epoch_metrics['ade'] += ade.item()
                epoch_metrics['fde'] += fde.item()
                
                num_batches += 1
        
        # Restore original model
        if self.ema is not None:
            self.ema.restore(self.model, backup)
        
        # Average metrics
        metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return metrics
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single forward step.
        
        Args:
            batch: Input batch
        
        Returns:
            Loss dictionary
        """
        # Get model output
        obs_traj = batch['obs_traj']
        pred_traj_gt = batch['pred_traj']
        obs_pose = batch.get('obs_pose')
        obs_velocity = batch.get('obs_velocity')
        
        # Model forward
        if hasattr(self.model, 'sample') and self.model.training:
            # Stochastic model - use sample during training for variety loss
            pred_traj = self.model.sample(
                obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_velocity,
                num_samples=self.config.get('model', {}).get('k_samples', 20)
            )
        else:
            # Deterministic model
            pred_traj = self.model(
                obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_velocity
            )
        
        # Compute loss
        loss_dict = self.loss_fn(
            pred_traj,
            pred_traj_gt,
            obs_velocity=obs_velocity
        )
        
        return loss_dict
    
    def _get_predictions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get model predictions for evaluation.
        
        Args:
            batch: Input batch
        
        Returns:
            Predicted trajectories [B, K, T, 2] or [B, T, 2]
        """
        obs_traj = batch['obs_traj']
        obs_pose = batch.get('obs_pose')
        obs_velocity = batch.get('obs_velocity')
        
        if hasattr(self.model, 'sample'):
            pred_traj = self.model.sample(
                obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_velocity,
                num_samples=self.config.get('model', {}).get('k_samples', 20)
            )
        else:
            pred_traj = self.model(
                obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_velocity
            )
        
        return pred_traj
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics."""
        # Console logging
        logger.info(
            f"Epoch {epoch + 1}/{self.epochs} - "
            f"Train Loss: {train_metrics.get('total', 0):.4f}, "
            f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
            f"Val ADE: {val_metrics.get('ade', 0):.4f}, "
            f"Val FDE: {val_metrics.get('fde', 0):.4f}, "
            f"LR: {get_current_lr(self.optimizer):.6f}"
        )
        
        # Save to history
        self.history['epoch'].append(epoch + 1)
        self.history['train_loss'].append(train_metrics.get('total', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_ade'].append(val_metrics.get('ade', 0))
        self.history['val_fde'].append(val_metrics.get('fde', 0))
        self.history['lr'].append(get_current_lr(self.optimizer))
        
        # TensorBoard logging
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.add_scalar('lr', get_current_lr(self.optimizer), epoch)
    
    def _check_improvement(self, val_metrics: Dict[str, float]) -> bool:
        """Check if validation improved."""
        val_ade = val_metrics.get('ade', float('inf'))
        val_loss = val_metrics.get('loss', float('inf'))
        
        improved = False
        
        if val_ade < self.best_val_ade - self.min_delta:
            self.best_val_ade = val_ade
            improved = True
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
        
        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return improved
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ade': self.best_val_ade,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': dict(self.history),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save periodic
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        torch.save(checkpoint, epoch_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint (ADE: {self.best_val_ade:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_ade = checkpoint['best_val_ade']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")


def create_trainer_from_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict
) -> Trainer:
    """
    Create trainer from configuration dictionary.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
    
    Returns:
        Configured Trainer
    """
    # Create optimizer
    optimizer = get_optimizer_from_config(model, config)
    
    # Create scheduler
    scheduler = get_scheduler_from_config(optimizer, config)
    
    # Create loss function
    loss_config = config.get('loss', {})
    loss_fn = TrajectoryLoss(
        ade_weight=loss_config.get('ade_weight', 1.0),
        fde_weight=loss_config.get('fde_weight', 1.0),
        variety_weight=loss_config.get('variety_weight', 0.5),
        collision_weight=loss_config.get('collision_weight', 0.0),
        speed_weight=loss_config.get('speed_weight', 0.0),
        use_best_of_k=loss_config.get('use_best_of_k', True),
        k=config.get('model', {}).get('k_samples', 20),
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=SchedulerWrapper(scheduler),
        loss_fn=loss_fn,
        device=device,
        config=config,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
    
    Returns:
        (trained_model, training_history)
    """
    trainer = create_trainer_from_config(model, train_loader, val_loader, config)
    history = trainer.train()
    
    # Load best checkpoint
    best_path = trainer.checkpoint_dir / 'checkpoint_best.pt'
    if best_path.exists():
        trainer.load_checkpoint(str(best_path))
    
    return trainer.model, history
