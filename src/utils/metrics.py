"""
Evaluation metrics for trajectory prediction.
ADE, FDE, collision rate, and other metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_ade(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mode: str = 'mean'
) -> torch.Tensor:
    """
    Compute Average Displacement Error (ADE).
    Mean L2 distance across all predicted timesteps.
    
    Args:
        pred: [batch, pred_len, 2] predicted trajectory
        gt: [batch, pred_len, 2] ground truth trajectory
        mode: 'mean' for average, 'sum' for total
    
    Returns:
        ADE value
    """
    # L2 distance at each timestep
    displacement = torch.norm(pred - gt, dim=-1)  # [batch, pred_len]
    
    if mode == 'mean':
        ade = displacement.mean()
    elif mode == 'sum':
        ade = displacement.sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return ade


def compute_fde(
    pred: torch.Tensor,
    gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute Final Displacement Error (FDE).
    L2 distance at the final predicted timestep.
    
    Args:
        pred: [batch, pred_len, 2] predicted trajectory
        gt: [batch, pred_len, 2] ground truth trajectory
    
    Returns:
        FDE value
    """
    # L2 distance at final timestep
    final_displacement = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)  # [batch]
    fde = final_displacement.mean()
    
    return fde


def compute_ade_fde(
    pred: torch.Tensor,
    gt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both ADE and FDE.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
    
    Returns:
        (ade, fde)
    """
    ade = compute_ade(pred, gt)
    fde = compute_fde(pred, gt)
    return ade, fde


def compute_ade_at_time(
    pred: torch.Tensor,
    gt: torch.Tensor,
    timesteps: List[float],
    fps: float = 2.5
) -> Dict[str, float]:
    """
    Compute ADE at specific time horizons.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        timesteps: List of time horizons in seconds
        fps: Frames per second
    
    Returns:
        Dictionary of {time: ade}
    """
    results = {}
    pred_len = pred.size(1)
    
    for t in timesteps:
        frame_idx = int(t * fps) - 1
        if frame_idx < pred_len and frame_idx >= 0:
            ade_t = compute_ade(pred[:, :frame_idx+1], gt[:, :frame_idx+1])
            results[f'ADE@{t}s'] = ade_t.item()
    
    return results


def compute_best_of_k(
    predictions: torch.Tensor,
    gt: torch.Tensor,
    k: int = 20
) -> Tuple[float, float]:
    """
    Compute Best-of-K ADE and FDE.
    Standard evaluation protocol: generate K samples, report best.
    
    Args:
        predictions: [K, batch, pred_len, 2] K trajectory samples
        gt: [batch, pred_len, 2] ground truth
        k: Number of samples
    
    Returns:
        (best_ade, best_fde)
    """
    K = predictions.size(0)
    batch_size = predictions.size(1)
    
    # Compute ADE and FDE for each sample
    ades = []
    fdes = []
    
    for i in range(K):
        ade = compute_ade(predictions[i], gt)
        fde = compute_fde(predictions[i], gt)
        ades.append(ade)
        fdes.append(fde)
    
    ades = torch.stack(ades)
    fdes = torch.stack(fdes)
    
    best_ade = ades.min().item()
    best_fde = fdes.min().item()
    
    return best_ade, best_fde


def compute_collision_rate(
    predictions: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute collision rate between predicted agent trajectories.
    
    Args:
        predictions: [batch, num_agents, pred_len, 2]
        threshold: Distance threshold for collision (meters)
    
    Returns:
        Collision rate (0 to 1)
    """
    batch_size, num_agents, pred_len, _ = predictions.shape
    
    if num_agents < 2:
        return 0.0
    
    collisions = 0
    total_pairs = 0
    
    for t in range(pred_len):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = torch.norm(
                    predictions[:, i, t] - predictions[:, j, t], dim=-1
                )
                collisions += (dist < threshold).sum().item()
                total_pairs += batch_size
    
    return collisions / total_pairs if total_pairs > 0 else 0.0


def compute_miss_rate(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    Compute miss rate (FDE > threshold).
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        threshold: Distance threshold (meters)
    
    Returns:
        Miss rate (0 to 1)
    """
    final_displacement = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)
    miss_rate = (final_displacement > threshold).float().mean().item()
    return miss_rate


class TrajectoryEvaluator:
    """
    Complete evaluation class for trajectory prediction.
    """
    
    def __init__(
        self,
        k_samples: int = 20,
        time_horizons: List[float] = [1.0, 2.0, 3.0, 4.8],
        fps: float = 2.5,
        collision_threshold: float = 0.5
    ):
        """
        Args:
            k_samples: Number of samples for best-of-K evaluation
            time_horizons: Time horizons for ADE@T evaluation
            fps: Frames per second
            collision_threshold: Threshold for collision detection
        """
        self.k_samples = k_samples
        self.time_horizons = time_horizons
        self.fps = fps
        self.collision_threshold = collision_threshold
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_ade = []
        self.all_fde = []
        self.all_ade_at_time = {f'ADE@{t}s': [] for t in self.time_horizons}
        self.num_samples = 0
    
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        multi_sample: bool = False
    ):
        """
        Update metrics with new batch.
        
        Args:
            pred: [batch, pred_len, 2] or [K, batch, pred_len, 2]
            gt: [batch, pred_len, 2]
            multi_sample: Whether pred contains multiple samples
        """
        if multi_sample:
            # Best-of-K evaluation
            best_ade, best_fde = compute_best_of_k(pred, gt, self.k_samples)
            self.all_ade.append(best_ade)
            self.all_fde.append(best_fde)
            
            # Use best sample for time-based metrics
            K = pred.size(0)
            per_sample_ade = []
            for i in range(K):
                per_sample_ade.append(compute_ade(pred[i], gt).item())
            best_idx = np.argmin(per_sample_ade)
            best_pred = pred[best_idx]
        else:
            ade = compute_ade(pred, gt).item()
            fde = compute_fde(pred, gt).item()
            self.all_ade.append(ade)
            self.all_fde.append(fde)
            best_pred = pred
        
        # Time-based metrics
        ade_at_time = compute_ade_at_time(best_pred, gt, self.time_horizons, self.fps)
        for key, value in ade_at_time.items():
            self.all_ade_at_time[key].append(value)
        
        self.num_samples += gt.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metrics
        """
        results = {
            'ADE': np.mean(self.all_ade) if self.all_ade else 0.0,
            'FDE': np.mean(self.all_fde) if self.all_fde else 0.0,
            'ADE_std': np.std(self.all_ade) if self.all_ade else 0.0,
            'FDE_std': np.std(self.all_fde) if self.all_fde else 0.0,
            'num_samples': self.num_samples
        }
        
        for key, values in self.all_ade_at_time.items():
            if values:
                results[key] = np.mean(values)
        
        return results
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        results = self.compute()
        return (f"ADE: {results['ADE']:.4f} ± {results['ADE_std']:.4f}, "
                f"FDE: {results['FDE']:.4f} ± {results['FDE_std']:.4f}")
