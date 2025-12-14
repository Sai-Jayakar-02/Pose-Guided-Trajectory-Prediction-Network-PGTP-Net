#!/usr/bin/env python3
"""
PGTP-Net Training Script
Pose-Guided Trajectory Prediction Network

Usage:
    python train.py --config configs/jta_3d_pose.yaml --output_dir outputs/jta_exp
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pgtpnet.models import PGTPNet
from pgtpnet.data import get_dataloader
from pgtpnet.training import Trainer, get_loss_function, get_optimizer, get_scheduler
from pgtpnet.utils import load_config, setup_logging, set_seed, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train PGTP-Net')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/experiment',
                        help='Output directory for checkpoints and logs')
    
    # Override config options
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    # Flags
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.optimizer.lr = args.lr
    if args.device is not None:
        config.hardware.device = args.device
    
    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        f.write(str(config))
    
    # Setup logging
    logger = setup_logging(output_dir / 'train.log')
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    
    # Device
    device = torch.device(config.hardware.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Data
    logger.info("Loading datasets...")
    train_loader = get_dataloader(config, split='train', debug=args.debug)
    val_loader = get_dataloader(config, split='val', debug=args.debug)
    logger.info(f"Train: {len(train_loader.dataset)} samples")
    logger.info(f"Val: {len(val_loader.dataset)} samples")
    
    # Model
    logger.info("Building model...")
    model = PGTPNet(config.model).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = get_loss_function(config.training.loss)
    optimizer = get_optimizer(model.parameters(), config.training.optimizer)
    scheduler = get_scheduler(optimizer, config.training.scheduler)
    
    # Resume from checkpoint
    start_epoch = 0
    best_ade = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_ade = checkpoint.get('best_ade', float('inf'))
    
    # Eval only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        logger.info(f"Validation - ADE: {val_metrics['ade']:.4f}, FDE: {val_metrics['fde']:.4f}")
        return
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    early_stopping_counter = 0
    
    for epoch in range(start_epoch, config.training.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        logger.info(
            f"Epoch [{epoch+1}/{config.training.epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val ADE: {val_metrics['ade']:.4f} | "
            f"Val FDE: {val_metrics['fde']:.4f}"
        )
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('ADE/train', train_metrics['ade'], epoch)
        writer.add_scalar('ADE/val', val_metrics['ade'], epoch)
        writer.add_scalar('FDE/val', val_metrics['fde'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['ade'] < best_ade
        if is_best:
            best_ade = val_metrics['ade']
            early_stopping_counter = 0
            logger.info(f"  ✓ New best model (ADE: {best_ade:.4f})")
        else:
            early_stopping_counter += 1
        
        # Save
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_ade': best_ade,
            'config': config,
        }
        
        torch.save(checkpoint, output_dir / 'latest.pt')
        
        if is_best:
            torch.save(checkpoint, output_dir / 'best.pt')
        
        if (epoch + 1) % config.training.save_every == 0:
            torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pt')
        
        # Early stopping
        if config.training.early_stopping.enabled:
            if early_stopping_counter >= config.training.early_stopping.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    writer.close()
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE - Best ADE: {best_ade:.4f}")
    logger.info("=" * 60)


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    ade_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch in pbar:
        # Move to device
        obs_traj = batch['obs_traj'].to(device)
        pred_traj = batch['pred_traj'].to(device)
        obs_pose = batch['obs_pose'].to(device) if 'obs_pose' in batch else None
        obs_vel = batch['obs_vel'].to(device) if 'obs_vel' in batch else None
        
        # Forward
        predictions, extras = model(
            obs_traj=obs_traj,
            obs_pose=obs_pose,
            obs_vel=obs_vel,
            gt_traj=pred_traj  # For CVAE training
        )
        
        # Loss
        loss_dict = criterion(predictions, pred_traj, extras.get('kl_loss'))
        loss = loss_dict['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), obs_traj.size(0))
        ade_meter.update(loss_dict['ade'].item(), obs_traj.size(0))
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'ade': f'{ade_meter.avg:.4f}'})
    
    return {'loss': loss_meter.avg, 'ade': ade_meter.avg}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config):
    """Evaluate model."""
    model.eval()
    
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()
    
    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        obs_traj = batch['obs_traj'].to(device)
        pred_traj = batch['pred_traj'].to(device)
        obs_pose = batch['obs_pose'].to(device) if 'obs_pose' in batch else None
        obs_vel = batch['obs_vel'].to(device) if 'obs_vel' in batch else None
        
        # Generate K samples
        predictions = model.sample(
            obs_traj=obs_traj,
            obs_pose=obs_pose,
            obs_vel=obs_vel,
            num_samples=config.evaluation.num_samples
        )
        # predictions: [B, K, T, 2]
        
        # Best-of-K evaluation
        B, K, T, _ = predictions.shape
        gt_expanded = pred_traj.unsqueeze(1).expand(-1, K, -1, -1)
        
        # ADE per sample
        ade_per_sample = torch.norm(predictions - gt_expanded, dim=-1).mean(dim=-1)  # [B, K]
        best_ade = ade_per_sample.min(dim=1)[0]  # [B]
        
        # FDE per sample
        fde_per_sample = torch.norm(predictions[:, :, -1] - gt_expanded[:, :, -1], dim=-1)  # [B, K]
        best_fde = fde_per_sample.min(dim=1)[0]  # [B]
        
        ade_meter.update(best_ade.mean().item(), B)
        fde_meter.update(best_fde.mean().item(), B)
    
    return {'ade': ade_meter.avg, 'fde': fde_meter.avg}


if __name__ == '__main__':
    main()
