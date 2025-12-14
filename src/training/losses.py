"""
Loss functions for trajectory prediction.

Standard metrics from trajectory prediction literature:
- ADE (Average Displacement Error): Mean L2 distance over all timesteps
- FDE (Final Displacement Error): L2 distance at final timestep
- Best-of-K: Minimum ADE/FDE over K samples (for stochastic models)
- Variety Loss: Encourages diversity in predictions

References:
- Social-LSTM (Alahi et al., 2016)
- Social-GAN (Gupta et al., 2018)
- Social-Pose (Gupta et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ADELoss(nn.Module):
    """
    Average Displacement Error (ADE) loss.
    
    ADE = (1/T) * sum_{t=1}^{T} ||pred_t - gt_t||_2
    
    This is the standard metric for trajectory prediction.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ADE loss.
        
        Args:
            pred: Predicted trajectories [B, T, 2] or [B, K, T, 2]
            target: Ground truth trajectories [B, T, 2]
            mask: Optional mask for valid timesteps [B, T]
        
        Returns:
            ADE loss value
        """
        # Handle multi-sample predictions [B, K, T, 2]
        if pred.dim() == 4:
            # Expand target to match: [B, K, T, 2]
            target = target.unsqueeze(1).expand_as(pred)
        
        # Compute L2 distance at each timestep
        # [B, T] or [B, K, T]
        displacement = torch.norm(pred - target, p=2, dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            if displacement.dim() == 3:  # [B, K, T]
                mask = mask.unsqueeze(1)  # [B, 1, T]
            displacement = displacement * mask
            # Average over valid timesteps
            ade = displacement.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        else:
            # Average over timesteps
            ade = displacement.mean(dim=-1)  # [B] or [B, K]
        
        # Reduction
        if self.reduction == 'mean':
            return ade.mean()
        elif self.reduction == 'sum':
            return ade.sum()
        else:
            return ade


class FDELoss(nn.Module):
    """
    Final Displacement Error (FDE) loss.
    
    FDE = ||pred_T - gt_T||_2
    
    Measures error at the final prediction timestep.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute FDE loss.
        
        Args:
            pred: Predicted trajectories [B, T, 2] or [B, K, T, 2]
            target: Ground truth trajectories [B, T, 2]
        
        Returns:
            FDE loss value
        """
        # Handle multi-sample predictions
        if pred.dim() == 4:
            # Get final timestep: [B, K, 2]
            pred_final = pred[:, :, -1, :]
            target_final = target[:, -1:, :].expand(-1, pred.size(1), -1)
        else:
            # Get final timestep: [B, 2]
            pred_final = pred[:, -1, :]
            target_final = target[:, -1, :]
        
        # Compute L2 distance
        fde = torch.norm(pred_final - target_final, p=2, dim=-1)
        
        # Reduction
        if self.reduction == 'mean':
            return fde.mean()
        elif self.reduction == 'sum':
            return fde.sum()
        else:
            return fde


class BestOfKLoss(nn.Module):
    """
    Best-of-K loss for stochastic trajectory prediction.
    
    Computes minimum ADE/FDE over K samples, encouraging
    at least one prediction to be close to ground truth.
    
    Used in Social-GAN, Trajectron++, etc.
    """
    
    def __init__(
        self,
        k: int = 20,
        metric: str = 'ade',
        reduction: str = 'mean'
    ):
        """
        Args:
            k: Number of samples
            metric: 'ade' or 'fde'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.k = k
        self.metric = metric
        self.reduction = reduction
        
        self.ade_loss = ADELoss(reduction='none')
        self.fde_loss = FDELoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Best-of-K loss.
        
        Args:
            pred: Predicted trajectories [B, K, T, 2]
            target: Ground truth trajectories [B, T, 2]
        
        Returns:
            Best-of-K loss value
        """
        assert pred.dim() == 4, f"Expected [B, K, T, 2], got {pred.shape}"
        
        # Compute metric for all K samples
        if self.metric == 'ade':
            errors = self.ade_loss(pred, target)  # [B, K]
        else:
            errors = self.fde_loss(pred, target)  # [B, K]
        
        # Take minimum over K samples
        best_errors, _ = errors.min(dim=-1)  # [B]
        
        # Reduction
        if self.reduction == 'mean':
            return best_errors.mean()
        elif self.reduction == 'sum':
            return best_errors.sum()
        else:
            return best_errors


class VarietyLoss(nn.Module):
    """
    Variety (Diversity) loss to encourage diverse predictions.
    
    Penalizes predictions that are too similar to each other.
    This is crucial for multi-modal trajectory prediction.
    
    variety = -mean(||pred_i - pred_j||_2) for i != j
    
    Reference: Social-GAN (Gupta et al., 2018)
    """
    
    def __init__(
        self,
        min_distance: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Args:
            min_distance: Target minimum distance between samples
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.min_distance = min_distance
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute variety loss.
        
        Args:
            pred: Predicted trajectories [B, K, T, 2]
        
        Returns:
            Variety loss (negative, to be minimized)
        """
        B, K, T, D = pred.shape
        
        if K < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Flatten temporal dimension: [B, K, T*D]
        pred_flat = pred.view(B, K, -1)
        
        # Compute pairwise distances using broadcasting
        # [B, K, 1, T*D] - [B, 1, K, T*D] -> [B, K, K, T*D]
        diff = pred_flat.unsqueeze(2) - pred_flat.unsqueeze(1)
        distances = torch.norm(diff, p=2, dim=-1)  # [B, K, K]
        
        # Mask diagonal (distance to self = 0)
        mask = ~torch.eye(K, dtype=torch.bool, device=pred.device)
        distances = distances[:, mask].view(B, -1)  # [B, K*(K-1)]
        
        # Average pairwise distance
        avg_distance = distances.mean(dim=-1)  # [B]
        
        # Loss: penalize if average distance is below threshold
        # We want to maximize distance, so minimize negative distance
        loss = F.relu(self.min_distance - avg_distance)
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class CollisionLoss(nn.Module):
    """
    Collision avoidance loss.
    
    Penalizes predictions that come too close to other pedestrians.
    Useful for social-aware trajectory prediction.
    """
    
    def __init__(
        self,
        collision_threshold: float = 0.2,
        reduction: str = 'mean'
    ):
        """
        Args:
            collision_threshold: Distance below which collision is penalized (meters)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.collision_threshold = collision_threshold
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        neighbors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute collision loss.
        
        Args:
            pred: Predicted trajectory [B, T, 2] or [B, K, T, 2]
            neighbors: Neighbor trajectories [B, N, T, 2]
        
        Returns:
            Collision loss
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [B, 1, T, 2]
        
        B, K, T, _ = pred.shape
        N = neighbors.size(1)
        
        if N == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Expand for pairwise comparison
        # pred: [B, K, 1, T, 2], neighbors: [B, 1, N, T, 2]
        pred_exp = pred.unsqueeze(2)
        neighbors_exp = neighbors.unsqueeze(1)
        
        # Compute distances: [B, K, N, T]
        distances = torch.norm(pred_exp - neighbors_exp, p=2, dim=-1)
        
        # Penalize distances below threshold
        collision_penalty = F.relu(self.collision_threshold - distances)
        
        # Average over all pairs and timesteps
        loss = collision_penalty.mean(dim=(-1, -2, -3))  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class SpeedConsistencyLoss(nn.Module):
    """
    Speed consistency loss for adaptive trajectory prediction.
    
    Encourages predicted speeds to be consistent with observed speeds,
    while allowing for natural acceleration/deceleration.
    """
    
    def __init__(
        self,
        max_acceleration: float = 2.0,  # m/s^2
        dt: float = 0.4,  # Time step at 2.5 FPS
        reduction: str = 'mean'
    ):
        """
        Args:
            max_acceleration: Maximum allowed acceleration
            dt: Time step between frames
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        obs_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute speed consistency loss.
        
        Args:
            pred: Predicted trajectory [B, T, 2]
            obs_velocity: Observed velocities [B, obs_len, 2]
        
        Returns:
            Speed consistency loss
        """
        # Compute predicted velocities
        pred_velocity = torch.diff(pred, dim=1) / self.dt  # [B, T-1, 2]
        
        # Last observed velocity
        last_obs_vel = obs_velocity[:, -1, :]  # [B, 2]
        
        # First predicted velocity
        first_pred_vel = pred_velocity[:, 0, :]  # [B, 2]
        
        # Acceleration at transition
        acceleration = (first_pred_vel - last_obs_vel) / self.dt
        accel_magnitude = torch.norm(acceleration, p=2, dim=-1)
        
        # Penalize excessive acceleration
        loss = F.relu(accel_magnitude - self.max_acceleration)
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for uncertainty estimation.
    
    Used when model predicts both mean and variance.
    """
    
    def __init__(
        self,
        min_var: float = 1e-4,
        reduction: str = 'mean'
    ):
        """
        Args:
            min_var: Minimum variance to prevent numerical instability
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.min_var = min_var
        self.reduction = reduction
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            pred_mean: Predicted mean [B, T, 2]
            pred_var: Predicted variance [B, T, 2] or [B, T, 1]
            target: Ground truth [B, T, 2]
        
        Returns:
            NLL loss
        """
        # Ensure minimum variance
        var = torch.clamp(pred_var, min=self.min_var)
        
        # NLL = 0.5 * (log(var) + (x - mu)^2 / var)
        diff_sq = (target - pred_mean) ** 2
        nll = 0.5 * (torch.log(var) + diff_sq / var)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction.
    
    Combines multiple loss components with configurable weights.
    """
    
    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 1.0,
        variety_weight: float = 0.5,
        collision_weight: float = 0.0,
        speed_weight: float = 0.0,
        use_best_of_k: bool = True,
        k: int = 20,
    ):
        """
        Args:
            ade_weight: Weight for ADE loss
            fde_weight: Weight for FDE loss
            variety_weight: Weight for variety loss
            collision_weight: Weight for collision loss
            speed_weight: Weight for speed consistency loss
            use_best_of_k: Use Best-of-K for ADE/FDE
            k: Number of samples for Best-of-K
        """
        super().__init__()
        
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.variety_weight = variety_weight
        self.collision_weight = collision_weight
        self.speed_weight = speed_weight
        self.use_best_of_k = use_best_of_k
        self.k = k
        
        # Loss components
        if use_best_of_k:
            self.ade_loss = BestOfKLoss(k=k, metric='ade')
            self.fde_loss = BestOfKLoss(k=k, metric='fde')
        else:
            self.ade_loss = ADELoss()
            self.fde_loss = FDELoss()
        
        self.variety_loss = VarietyLoss() if variety_weight > 0 else None
        self.collision_loss = CollisionLoss() if collision_weight > 0 else None
        self.speed_loss = SpeedConsistencyLoss() if speed_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        extras: Optional[Dict] = None,
        neighbors: Optional[torch.Tensor] = None,
        obs_velocity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted trajectories [B, T, 2] or [B, K, T, 2]
            target: Ground truth trajectories [B, T, 2]
            extras: Optional dict with 'kl_loss', 'speeds', etc. from model
            neighbors: Neighbor trajectories [B, N, T, 2] (optional)
            obs_velocity: Observed velocities [B, obs_len, 2] (optional)
        
        Returns:
            Dictionary with 'total' loss and individual components
        """
        losses = {}
        
        # Handle both 3D [B, T, 2] and 4D [B, K, T, 2] inputs
        if pred.dim() == 3:
            # Single sample - use simple ADE/FDE
            # Displacement errors
            displacement = pred - target  # [B, T, 2]
            l2_errors = torch.norm(displacement, dim=-1)  # [B, T]
            
            # ADE: mean over time
            losses['ade'] = l2_errors.mean() * self.ade_weight
            
            # FDE: error at final timestep
            losses['fde'] = l2_errors[:, -1].mean() * self.fde_weight
        else:
            # Multi-sample - use Best-of-K
            losses['ade'] = self.ade_loss(pred, target) * self.ade_weight
            losses['fde'] = self.fde_loss(pred, target) * self.fde_weight
            
            # Variety (only for multi-sample predictions)
            if self.variety_loss is not None:
                losses['variety'] = self.variety_loss(pred) * self.variety_weight
        
        # Add KL loss from CVAE if present
        if extras is not None and 'kl_loss' in extras:
            kl_weight = 0.1  # Beta for beta-VAE
            losses['kl'] = extras['kl_loss'] * kl_weight
        
        # Collision
        if self.collision_loss is not None and neighbors is not None:
            losses['collision'] = self.collision_loss(pred, neighbors) * self.collision_weight
        
        # Speed consistency
        if self.speed_loss is not None and obs_velocity is not None:
            # Use mean prediction for speed loss
            pred_mean = pred.mean(dim=1) if pred.dim() == 4 else pred
            losses['speed'] = self.speed_loss(pred_mean, obs_velocity) * self.speed_weight
        
        # Total loss
        losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        
        return losses


# =============================================================================
# Utility Functions
# =============================================================================

def compute_ade(
    pred: torch.Tensor,
    target: torch.Tensor,
    best_of_k: bool = True
) -> torch.Tensor:
    """
    Compute ADE metric (for evaluation, not training).
    
    Args:
        pred: Predictions [B, T, 2] or [B, K, T, 2]
        target: Ground truth [B, T, 2]
        best_of_k: If True and pred has K samples, return min ADE
    
    Returns:
        ADE value
    """
    with torch.no_grad():
        if pred.dim() == 4 and best_of_k:
            loss_fn = BestOfKLoss(k=pred.size(1), metric='ade')
        else:
            loss_fn = ADELoss()
        return loss_fn(pred, target)


def compute_fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    best_of_k: bool = True
) -> torch.Tensor:
    """
    Compute FDE metric (for evaluation, not training).
    
    Args:
        pred: Predictions [B, T, 2] or [B, K, T, 2]
        target: Ground truth [B, T, 2]
        best_of_k: If True and pred has K samples, return min FDE
    
    Returns:
        FDE value
    """
    with torch.no_grad():
        if pred.dim() == 4 and best_of_k:
            loss_fn = BestOfKLoss(k=pred.size(1), metric='fde')
        else:
            loss_fn = FDELoss()
        return loss_fn(pred, target)


def compute_collision_rate(
    pred: torch.Tensor,
    neighbors: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Compute collision rate metric.
    
    Args:
        pred: Predictions [B, T, 2] or [B, K, T, 2]
        neighbors: Neighbor positions [B, N, T, 2]
        threshold: Collision distance threshold
    
    Returns:
        Collision rate (fraction of samples with collision)
    """
    with torch.no_grad():
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        
        B, K, T, _ = pred.shape
        N = neighbors.size(1)
        
        if N == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute distances
        pred_exp = pred.unsqueeze(2)  # [B, K, 1, T, 2]
        neighbors_exp = neighbors.unsqueeze(1)  # [B, 1, N, T, 2]
        distances = torch.norm(pred_exp - neighbors_exp, p=2, dim=-1)  # [B, K, N, T]
        
        # Check for collisions
        min_distances = distances.min(dim=-1)[0].min(dim=-1)[0]  # [B, K]
        collisions = (min_distances < threshold).float()
        
        return collisions.mean()