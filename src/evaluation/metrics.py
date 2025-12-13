"""
Extended Evaluation Metrics for Trajectory Prediction.

Standard metrics:
- ADE (Average Displacement Error): Mean L2 distance across all timesteps
- FDE (Final Displacement Error): L2 distance at final timestep
- Best-of-K: Minimum ADE/FDE over K stochastic samples

Extended metrics:
- ADE@T: ADE at specific time horizons (1s, 2s, 3s, 4.8s)
- Miss Rate: Percentage of FDE > threshold
- Collision Rate: Percentage of predicted trajectories with collisions
- NLL (Negative Log-Likelihood): For probabilistic models
- KDE (Kernel Density Estimation): Distribution quality
- Diversity: Variance across predicted samples

References:
- ETH/UCY benchmark: Pellegrini et al. (2009), Lerner et al. (2007)
- Social-GAN evaluation: Gupta et al. (2018)
- Trajectron++ metrics: Salzmann et al. (2020)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import gaussian_kde
import warnings


# =============================================================================
# Core Metrics (ADE, FDE)
# =============================================================================

def compute_ade(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = 'mean'
) -> torch.Tensor:
    """
    Compute Average Displacement Error (ADE).
    
    ADE = (1/T) * sum_{t=1}^{T} ||pred_t - gt_t||_2
    
    Args:
        pred: Predicted trajectory [batch, pred_len, 2] or [batch, pred_len, 3]
        gt: Ground truth trajectory [batch, pred_len, 2] or [batch, pred_len, 3]
        mask: Optional validity mask [batch, pred_len]
        mode: 'mean' for average, 'sum' for total, 'none' for per-sample
    
    Returns:
        ADE value(s)
    """
    # L2 distance at each timestep
    displacement = torch.norm(pred - gt, p=2, dim=-1)  # [batch, pred_len]
    
    if mask is not None:
        displacement = displacement * mask
        valid_counts = mask.sum(dim=-1).clamp(min=1)
        ade_per_sample = displacement.sum(dim=-1) / valid_counts
    else:
        ade_per_sample = displacement.mean(dim=-1)  # [batch]
    
    if mode == 'mean':
        return ade_per_sample.mean()
    elif mode == 'sum':
        return ade_per_sample.sum()
    elif mode == 'none':
        return ade_per_sample
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_fde(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mode: str = 'mean'
) -> torch.Tensor:
    """
    Compute Final Displacement Error (FDE).
    
    FDE = ||pred_T - gt_T||_2
    
    Args:
        pred: Predicted trajectory [batch, pred_len, 2]
        gt: Ground truth trajectory [batch, pred_len, 2]
        mode: 'mean', 'sum', or 'none'
    
    Returns:
        FDE value(s)
    """
    # L2 distance at final timestep
    fde_per_sample = torch.norm(pred[:, -1] - gt[:, -1], p=2, dim=-1)  # [batch]
    
    if mode == 'mean':
        return fde_per_sample.mean()
    elif mode == 'sum':
        return fde_per_sample.sum()
    elif mode == 'none':
        return fde_per_sample
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_ade_fde(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both ADE and FDE.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        mask: Optional [batch, pred_len]
    
    Returns:
        (ade, fde) tuple
    """
    ade = compute_ade(pred, gt, mask=mask)
    fde = compute_fde(pred, gt)
    return ade, fde


# =============================================================================
# Best-of-K Metrics (for stochastic models)
# =============================================================================

def compute_best_of_k_ade(
    predictions: torch.Tensor,
    gt: torch.Tensor,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Best-of-K ADE (minADE_K).
    
    For each sample, generate K predictions and report minimum ADE.
    
    Args:
        predictions: [K, batch, pred_len, 2] or [batch, K, pred_len, 2]
        gt: [batch, pred_len, 2]
        k: Number of samples (uses all if None)
    
    Returns:
        Best ADE value
    """
    # Ensure shape is [K, batch, pred_len, 2]
    if predictions.dim() == 4:
        if predictions.size(0) != gt.size(0):
            # Shape is [K, batch, pred_len, 2]
            pass
        else:
            # Shape is [batch, K, pred_len, 2] -> transpose
            predictions = predictions.permute(1, 0, 2, 3)
    
    K = predictions.size(0)
    if k is not None:
        K = min(k, K)
        predictions = predictions[:K]
    
    # Compute ADE for each sample
    ade_per_k = []
    for i in range(K):
        ade = compute_ade(predictions[i], gt, mode='none')  # [batch]
        ade_per_k.append(ade)
    
    ade_per_k = torch.stack(ade_per_k, dim=0)  # [K, batch]
    
    # Best (minimum) ADE per batch element
    best_ade = ade_per_k.min(dim=0)[0]  # [batch]
    
    return best_ade.mean()


def compute_best_of_k_fde(
    predictions: torch.Tensor,
    gt: torch.Tensor,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Best-of-K FDE (minFDE_K).
    
    Args:
        predictions: [K, batch, pred_len, 2] or [batch, K, pred_len, 2]
        gt: [batch, pred_len, 2]
        k: Number of samples
    
    Returns:
        Best FDE value
    """
    # Ensure shape is [K, batch, pred_len, 2]
    if predictions.dim() == 4:
        if predictions.size(0) != gt.size(0):
            pass
        else:
            predictions = predictions.permute(1, 0, 2, 3)
    
    K = predictions.size(0)
    if k is not None:
        K = min(k, K)
        predictions = predictions[:K]
    
    # Compute FDE for each sample
    fde_per_k = []
    for i in range(K):
        fde = compute_fde(predictions[i], gt, mode='none')  # [batch]
        fde_per_k.append(fde)
    
    fde_per_k = torch.stack(fde_per_k, dim=0)  # [K, batch]
    
    # Best (minimum) FDE per batch element
    best_fde = fde_per_k.min(dim=0)[0]  # [batch]
    
    return best_fde.mean()


def compute_best_of_k(
    predictions: torch.Tensor,
    gt: torch.Tensor,
    k: int = 20
) -> Tuple[float, float]:
    """
    Compute both Best-of-K ADE and FDE.
    
    Standard evaluation protocol for stochastic models:
    Generate K samples, report best (minimum) ADE and FDE.
    
    Args:
        predictions: [K, batch, pred_len, 2] K trajectory samples
        gt: [batch, pred_len, 2] ground truth
        k: Number of samples
    
    Returns:
        (best_ade, best_fde) tuple
    """
    best_ade = compute_best_of_k_ade(predictions, gt, k)
    best_fde = compute_best_of_k_fde(predictions, gt, k)
    
    return best_ade.item(), best_fde.item()


# =============================================================================
# Time-Horizon Metrics
# =============================================================================

def compute_ade_at_time(
    pred: torch.Tensor,
    gt: torch.Tensor,
    time_seconds: List[float],
    fps: float = 2.5
) -> Dict[str, float]:
    """
    Compute ADE at specific time horizons.
    
    Common horizons: 1.0s, 2.0s, 3.0s, 4.8s (full 12 frames at 2.5 FPS)
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        time_seconds: List of time horizons in seconds
        fps: Frames per second (2.5 for ETH/UCY standard)
    
    Returns:
        Dictionary of {f'ADE@{t}s': value}
    """
    results = {}
    pred_len = pred.size(1)
    
    for t in time_seconds:
        # Convert seconds to frame index
        frame_idx = int(t * fps)
        
        if frame_idx > 0 and frame_idx <= pred_len:
            # ADE up to this time horizon
            ade_t = compute_ade(pred[:, :frame_idx], gt[:, :frame_idx])
            results[f'ADE@{t}s'] = ade_t.item()
    
    return results


def compute_fde_at_time(
    pred: torch.Tensor,
    gt: torch.Tensor,
    time_seconds: List[float],
    fps: float = 2.5
) -> Dict[str, float]:
    """
    Compute FDE at specific time horizons.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        time_seconds: List of time horizons
        fps: Frames per second
    
    Returns:
        Dictionary of {f'FDE@{t}s': value}
    """
    results = {}
    pred_len = pred.size(1)
    
    for t in time_seconds:
        frame_idx = int(t * fps)
        
        if frame_idx > 0 and frame_idx <= pred_len:
            # FDE at this specific time
            fde_t = torch.norm(
                pred[:, frame_idx - 1] - gt[:, frame_idx - 1], 
                p=2, dim=-1
            ).mean()
            results[f'FDE@{t}s'] = fde_t.item()
    
    return results


# =============================================================================
# Miss Rate and Collision Metrics
# =============================================================================

def compute_miss_rate(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    Compute miss rate (percentage of FDE > threshold).
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        threshold: Distance threshold in meters (default 2.0m)
    
    Returns:
        Miss rate (0.0 to 1.0)
    """
    fde = compute_fde(pred, gt, mode='none')  # [batch]
    miss_rate = (fde > threshold).float().mean().item()
    return miss_rate


def compute_collision_rate(
    predictions: torch.Tensor,
    threshold: float = 0.1
) -> float:
    """
    Compute collision rate between multiple agents.
    
    A collision occurs when two agents are within threshold distance.
    
    Args:
        predictions: [batch, num_agents, pred_len, 2]
        threshold: Collision distance threshold in meters
    
    Returns:
        Collision rate (0.0 to 1.0)
    """
    batch_size, num_agents, pred_len, _ = predictions.shape
    
    if num_agents < 2:
        return 0.0
    
    total_collisions = 0
    total_pairs = 0
    
    for b in range(batch_size):
        for t in range(pred_len):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = torch.norm(
                        predictions[b, i, t] - predictions[b, j, t],
                        p=2
                    )
                    if dist < threshold:
                        total_collisions += 1
                    total_pairs += 1
    
    return total_collisions / max(total_pairs, 1)


def compute_self_collision_rate(
    predictions: torch.Tensor,
    min_distance: float = 0.1
) -> float:
    """
    Compute self-collision rate (trajectory crosses itself).
    
    Args:
        predictions: [batch, pred_len, 2]
        min_distance: Minimum allowed distance between non-adjacent points
    
    Returns:
        Self-collision rate
    """
    batch_size, pred_len, _ = predictions.shape
    
    collisions = 0
    total = 0
    
    for b in range(batch_size):
        for i in range(pred_len):
            for j in range(i + 2, pred_len):  # Skip adjacent points
                dist = torch.norm(predictions[b, i] - predictions[b, j], p=2)
                if dist < min_distance:
                    collisions += 1
                total += 1
    
    return collisions / max(total, 1)


# =============================================================================
# Probabilistic Metrics
# =============================================================================

def compute_nll(
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Negative Log-Likelihood for Gaussian predictions.
    
    NLL = 0.5 * (log(2π) + log(σ²) + (y - μ)² / σ²)
    
    Args:
        pred_mean: Predicted mean [batch, pred_len, 2]
        pred_var: Predicted variance [batch, pred_len, 2]
        gt: Ground truth [batch, pred_len, 2]
        eps: Small constant for numerical stability
    
    Returns:
        NLL value
    """
    pred_var = pred_var.clamp(min=eps)
    
    nll = 0.5 * (
        np.log(2 * np.pi) +
        torch.log(pred_var) +
        (gt - pred_mean) ** 2 / pred_var
    )
    
    return nll.mean()


def compute_kde_nll(
    predictions: torch.Tensor,
    gt: torch.Tensor,
    bandwidth: float = 0.1
) -> float:
    """
    Compute NLL using Kernel Density Estimation.
    
    Used when model outputs samples rather than parametric distribution.
    
    Args:
        predictions: [K, batch, pred_len, 2] K samples
        gt: [batch, pred_len, 2] ground truth
        bandwidth: KDE bandwidth
    
    Returns:
        KDE-based NLL
    """
    K, batch_size, pred_len, _ = predictions.shape
    
    nll_total = 0.0
    count = 0
    
    for b in range(batch_size):
        for t in range(pred_len):
            # Get samples for this point
            samples = predictions[:, b, t, :].cpu().numpy()  # [K, 2]
            gt_point = gt[b, t, :].cpu().numpy()  # [2]
            
            try:
                # Fit KDE
                kde = gaussian_kde(samples.T, bw_method=bandwidth)
                # Evaluate at ground truth
                log_prob = np.log(kde(gt_point) + 1e-10)
                nll_total -= log_prob[0]
                count += 1
            except Exception:
                # KDE can fail if samples are degenerate
                continue
    
    return nll_total / max(count, 1)


# =============================================================================
# Diversity Metrics
# =============================================================================

def compute_diversity(
    predictions: torch.Tensor,
    mode: str = 'mean'
) -> float:
    """
    Compute diversity of predicted trajectories.
    
    Diversity = average pairwise distance between samples
    
    Args:
        predictions: [K, batch, pred_len, 2] or [batch, K, pred_len, 2]
        mode: 'mean' for average, 'final' for final positions only
    
    Returns:
        Diversity value
    """
    # Ensure [K, batch, pred_len, 2]
    if predictions.dim() == 4:
        if predictions.size(1) > predictions.size(0):
            predictions = predictions.permute(1, 0, 2, 3)
    
    K = predictions.size(0)
    
    if K < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    
    for i in range(K):
        for j in range(i + 1, K):
            if mode == 'final':
                # Distance between final positions
                dist = torch.norm(
                    predictions[i, :, -1] - predictions[j, :, -1],
                    p=2, dim=-1
                ).mean()
            else:
                # Average distance across all timesteps
                dist = torch.norm(
                    predictions[i] - predictions[j],
                    p=2, dim=-1
                ).mean()
            
            total_dist += dist.item()
            count += 1
    
    return total_dist / max(count, 1)


def compute_sample_variance(
    predictions: torch.Tensor
) -> float:
    """
    Compute variance across predicted samples.
    
    Args:
        predictions: [K, batch, pred_len, 2]
    
    Returns:
        Average variance
    """
    # Variance across K dimension
    var = predictions.var(dim=0)  # [batch, pred_len, 2]
    return var.mean().item()


# =============================================================================
# Speed and Motion Metrics
# =============================================================================

def compute_speed_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    dt: float = 0.4
) -> float:
    """
    Compute error in predicted speed.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
        dt: Time step in seconds
    
    Returns:
        Speed error (m/s)
    """
    # Compute velocities
    pred_vel = (pred[:, 1:] - pred[:, :-1]) / dt
    gt_vel = (gt[:, 1:] - gt[:, :-1]) / dt
    
    # Compute speeds
    pred_speed = torch.norm(pred_vel, p=2, dim=-1)
    gt_speed = torch.norm(gt_vel, p=2, dim=-1)
    
    # Speed error
    speed_error = torch.abs(pred_speed - gt_speed).mean()
    
    return speed_error.item()


def compute_heading_error(
    pred: torch.Tensor,
    gt: torch.Tensor
) -> float:
    """
    Compute error in predicted heading direction.
    
    Args:
        pred: [batch, pred_len, 2]
        gt: [batch, pred_len, 2]
    
    Returns:
        Heading error in radians
    """
    # Compute displacements
    pred_disp = pred[:, 1:] - pred[:, :-1]
    gt_disp = gt[:, 1:] - gt[:, :-1]
    
    # Compute headings
    pred_heading = torch.atan2(pred_disp[:, :, 1], pred_disp[:, :, 0])
    gt_heading = torch.atan2(gt_disp[:, :, 1], gt_disp[:, :, 0])
    
    # Angular difference
    heading_error = torch.abs(pred_heading - gt_heading)
    # Wrap to [0, pi]
    heading_error = torch.min(heading_error, 2 * np.pi - heading_error)
    
    return heading_error.mean().item()


# =============================================================================
# Aggregated Metrics Class
# =============================================================================

class TrajectoryMetrics:
    """
    Comprehensive trajectory prediction metrics.
    
    Computes and aggregates all standard metrics for trajectory prediction.
    
    Usage:
        metrics = TrajectoryMetrics(k_samples=20)
        
        for batch in dataloader:
            pred = model.sample(obs, num_samples=20)
            metrics.update(pred, gt)
        
        results = metrics.compute()
        print(results)
    """
    
    def __init__(
        self,
        k_samples: int = 20,
        time_horizons: List[float] = [1.0, 2.0, 3.0, 4.8],
        fps: float = 2.5,
        miss_threshold: float = 2.0,
        collision_threshold: float = 0.1,
    ):
        """
        Args:
            k_samples: Number of samples for Best-of-K
            time_horizons: Time horizons for ADE@T metrics
            fps: Dataset FPS
            miss_threshold: Threshold for miss rate
            collision_threshold: Threshold for collision rate
        """
        self.k_samples = k_samples
        self.time_horizons = time_horizons
        self.fps = fps
        self.miss_threshold = miss_threshold
        self.collision_threshold = collision_threshold
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_ade = []
        self.all_fde = []
        self.all_best_ade = []
        self.all_best_fde = []
        self.all_ade_at_time = {f'ADE@{t}s': [] for t in self.time_horizons}
        self.all_fde_at_time = {f'FDE@{t}s': [] for t in self.time_horizons}
        self.all_miss_rate = []
        self.all_diversity = []
        self.num_samples = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        gt: torch.Tensor,
        multi_sample: bool = True
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions: [K, batch, pred_len, 2] or [batch, pred_len, 2]
            gt: [batch, pred_len, 2]
            multi_sample: Whether predictions contains K samples
        """
        if multi_sample:
            # Best-of-K metrics
            best_ade, best_fde = compute_best_of_k(predictions, gt, self.k_samples)
            self.all_best_ade.append(best_ade)
            self.all_best_fde.append(best_fde)
            
            # Use best sample for other metrics
            K = predictions.size(0) if predictions.size(0) != gt.size(0) else predictions.size(1)
            if predictions.size(0) == gt.size(0):
                predictions = predictions.permute(1, 0, 2, 3)
            
            # Find best sample per batch element
            ade_per_k = torch.stack([
                compute_ade(predictions[k], gt, mode='none')
                for k in range(min(K, self.k_samples))
            ], dim=0)
            best_idx = ade_per_k.argmin(dim=0)
            
            # Get best predictions
            batch_size = gt.size(0)
            best_pred = torch.stack([
                predictions[best_idx[b], b] 
                for b in range(batch_size)
            ], dim=0)
            
            # Diversity
            diversity = compute_diversity(predictions)
            self.all_diversity.append(diversity)
        else:
            # Single prediction
            ade = compute_ade(predictions, gt).item()
            fde = compute_fde(predictions, gt).item()
            self.all_ade.append(ade)
            self.all_fde.append(fde)
            best_pred = predictions
        
        # Time-horizon metrics
        ade_at_time = compute_ade_at_time(best_pred, gt, self.time_horizons, self.fps)
        for key, value in ade_at_time.items():
            self.all_ade_at_time[key].append(value)
        
        fde_at_time = compute_fde_at_time(best_pred, gt, self.time_horizons, self.fps)
        for key, value in fde_at_time.items():
            self.all_fde_at_time[key].append(value)
        
        # Miss rate
        miss_rate = compute_miss_rate(best_pred, gt, self.miss_threshold)
        self.all_miss_rate.append(miss_rate)
        
        self.num_samples += gt.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics.
        
        Returns:
            Dictionary of all metrics
        """
        results = {}
        
        # Basic ADE/FDE
        if self.all_ade:
            results['ADE'] = np.mean(self.all_ade)
            results['FDE'] = np.mean(self.all_fde)
        
        # Best-of-K
        if self.all_best_ade:
            results[f'minADE_{self.k_samples}'] = np.mean(self.all_best_ade)
            results[f'minFDE_{self.k_samples}'] = np.mean(self.all_best_fde)
        
        # Time horizons
        for key, values in self.all_ade_at_time.items():
            if values:
                results[key] = np.mean(values)
        
        for key, values in self.all_fde_at_time.items():
            if values:
                results[key] = np.mean(values)
        
        # Miss rate
        if self.all_miss_rate:
            results[f'MissRate@{self.miss_threshold}m'] = np.mean(self.all_miss_rate)
        
        # Diversity
        if self.all_diversity:
            results['Diversity'] = np.mean(self.all_diversity)
        
        results['num_samples'] = self.num_samples
        
        return results
    
    def __str__(self) -> str:
        """String representation."""
        results = self.compute()
        
        lines = ["Trajectory Prediction Metrics:"]
        lines.append("-" * 40)
        
        if f'minADE_{self.k_samples}' in results:
            lines.append(f"  minADE_{self.k_samples}: {results[f'minADE_{self.k_samples}']:.4f}")
            lines.append(f"  minFDE_{self.k_samples}: {results[f'minFDE_{self.k_samples}']:.4f}")
        elif 'ADE' in results:
            lines.append(f"  ADE: {results['ADE']:.4f}")
            lines.append(f"  FDE: {results['FDE']:.4f}")
        
        if 'Diversity' in results:
            lines.append(f"  Diversity: {results['Diversity']:.4f}")
        
        lines.append(f"  Samples: {results['num_samples']}")
        
        return "\n".join(lines)
