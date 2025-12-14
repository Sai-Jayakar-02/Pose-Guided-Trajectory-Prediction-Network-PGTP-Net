#!/usr/bin/env python3
"""
PGTP-Net Evaluation Script
Evaluate trained models on test sets

Usage:
    python evaluate.py --model_path models/pgtp_jta_best.pt --dataset jta
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from pgtpnet.models import PGTPNet
from pgtpnet.data import get_dataloader
from pgtpnet.utils import load_config, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PGTP-Net')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config (uses saved config if not provided)')
    parser.add_argument('--dataset', type=str, default='jta',
                        choices=['jta', 'eth', 'hotel', 'univ', 'zara1', 'zara2'],
                        help='Dataset to evaluate on')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples for Best-of-K evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save results to file')
    
    return parser.parse_args()


def compute_metrics(predictions, ground_truth, num_samples=20):
    """
    Compute trajectory prediction metrics.
    
    Args:
        predictions: [B, K, T, 2] - K predicted trajectories
        ground_truth: [B, T, 2] - Ground truth trajectories
        num_samples: K value for Best-of-K
        
    Returns:
        Dictionary with ADE, FDE, minADE, minFDE
    """
    B, K, T, _ = predictions.shape
    gt_expanded = ground_truth.unsqueeze(1).expand(-1, K, -1, -1)
    
    # Per-sample errors
    displacement = torch.norm(predictions - gt_expanded, dim=-1)  # [B, K, T]
    
    # ADE: Average Displacement Error
    ade_per_sample = displacement.mean(dim=-1)  # [B, K]
    
    # FDE: Final Displacement Error
    fde_per_sample = displacement[:, :, -1]  # [B, K]
    
    # Best-of-K (minimum across K samples)
    min_ade = ade_per_sample.min(dim=1)[0]  # [B]
    min_fde = fde_per_sample.min(dim=1)[0]  # [B]
    
    # Mean ADE/FDE (average across K samples, then across batch)
    mean_ade = ade_per_sample.mean(dim=1)  # [B]
    mean_fde = fde_per_sample.mean(dim=1)  # [B]
    
    return {
        'ade': mean_ade.mean().item(),
        'fde': mean_fde.mean().item(),
        'min_ade': min_ade.mean().item(),
        'min_fde': min_fde.mean().item(),
        'ade_std': mean_ade.std().item(),
        'fde_std': mean_fde.std().item(),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, num_samples=20):
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        obs_traj = batch['obs_traj'].to(device)
        pred_traj = batch['pred_traj'].to(device)
        obs_pose = batch['obs_pose'].to(device) if 'obs_pose' in batch else None
        obs_vel = batch['obs_vel'].to(device) if 'obs_vel' in batch else None
        
        # Generate K samples
        predictions = model.sample(
            obs_traj=obs_traj,
            obs_pose=obs_pose,
            obs_vel=obs_vel,
            num_samples=num_samples
        )
        
        all_predictions.append(predictions.cpu())
        all_ground_truth.append(pred_traj.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_ground_truth, num_samples)
    
    return metrics, all_predictions, all_ground_truth


def main():
    args = parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config')
        if config is None:
            raise ValueError("No config found in checkpoint. Please provide --config")
    
    # Build model
    model = PGTPNet(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Get dataloader
    print(f"Loading {args.dataset} dataset...")
    
    if args.dataset == 'jta':
        dataloader = get_dataloader(config, split='test')
    else:
        # ETH/UCY - specific scene
        config.data.test_scene = args.dataset
        dataloader = get_dataloader(config, split='test')
    
    print(f"Test samples: {len(dataloader.dataset)}")
    
    # Evaluate
    print(f"\nEvaluating with K={args.num_samples} samples...")
    metrics, predictions, ground_truth = evaluate(
        model, dataloader, device, args.num_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - {args.dataset.upper()}")
    print("=" * 60)
    print(f"  ADE (mean):      {metrics['ade']:.4f} m")
    print(f"  FDE (mean):      {metrics['fde']:.4f} m")
    print(f"  minADE (K={args.num_samples}):  {metrics['min_ade']:.4f} m")
    print(f"  minFDE (K={args.num_samples}):  {metrics['min_fde']:.4f} m")
    print("=" * 60)
    
    # Save results
    if args.save_results:
        results = {
            'dataset': args.dataset,
            'model_path': args.model_path,
            'num_samples': args.num_samples,
            'metrics': metrics,
        }
        torch.save(results, args.save_results)
        print(f"Results saved to {args.save_results}")
    
    # Visualize
    if args.visualize:
        from pgtpnet.utils.visualization import visualize_predictions
        visualize_predictions(
            predictions[:10],
            ground_truth[:10],
            save_path=f'outputs/vis_{args.dataset}.png'
        )


if __name__ == '__main__':
    main()
