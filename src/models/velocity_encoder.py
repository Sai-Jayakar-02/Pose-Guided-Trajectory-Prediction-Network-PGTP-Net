"""
Velocity Encoder for speed-adaptive trajectory prediction.
Explicitly encodes speed and heading direction for adaptive trajectory spacing.
"""

import torch
import torch.nn as nn
from typing import Tuple


class VelocityEncoder(nn.Module):
    """
    Encode velocity information for speed-adaptive prediction.
    
    WHY VELOCITY ENCODER:
    - Speed determines trajectory point spacing
    - Heading direction indicates movement direction
    - Acceleration patterns indicate intent changes
    
    Input: Velocity vectors [batch, obs_len-1, 2]
    Output: Velocity features [batch, hidden_dim]
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64
    ):
        """
        Args:
            input_dim: Dimension of velocity vectors (2 for x,y)
            hidden_dim: Output hidden dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Process raw velocity
        self.velocity_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Speed embedding (magnitude)
        self.speed_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Heading embedding (direction as sin/cos)
        self.heading_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # sin(θ), cos(θ)
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Combine all velocity features
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Temporal aggregation with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
    
    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Encode velocity sequence.
        
        Args:
            velocity: [batch, seq_len, 2] velocity vectors
        
        Returns:
            features: [batch, hidden_dim]
        """
        batch_size, seq_len, _ = velocity.shape
        
        # Compute speed (magnitude)
        speed = torch.norm(velocity, dim=-1, keepdim=True)  # [B, T, 1]
        
        # Compute heading direction (normalized velocity)
        heading = velocity / (speed + 1e-8)  # [B, T, 2]
        
        # Embed each component
        vel_feat = self.velocity_embed(velocity)      # [B, T, hidden]
        speed_feat = self.speed_embed(speed)          # [B, T, hidden//2]
        heading_feat = self.heading_embed(heading)    # [B, T, hidden//2]
        
        # Combine features
        combined = torch.cat([vel_feat, speed_feat, heading_feat], dim=-1)
        combined = self.combine(combined)  # [B, T, hidden]
        
        # Temporal aggregation
        _, (hidden, _) = self.lstm(combined)
        
        return hidden.squeeze(0)  # [B, hidden]
    
    def get_speed_features(self, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get speed and heading features separately.
        
        Args:
            velocity: [batch, seq_len, 2]
        
        Returns:
            speeds: [batch, seq_len]
            headings: [batch, seq_len]
        """
        speed = torch.norm(velocity, dim=-1)  # [B, T]
        heading = torch.atan2(velocity[:, :, 1], velocity[:, :, 0])  # [B, T]
        
        return speed, heading


class AccelerationEncoder(nn.Module):
    """
    Encode acceleration for detecting changes in motion.
    Acceleration indicates intent changes (speeding up, slowing down, turning).
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        
        self.accel_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
    
    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Encode acceleration from velocity.
        
        Args:
            velocity: [batch, seq_len, 2]
        
        Returns:
            features: [batch, hidden_dim]
        """
        # Compute acceleration
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-1, 2]
        
        # Embed
        embedded = self.accel_embed(acceleration)
        
        # Aggregate
        _, (hidden, _) = self.lstm(embedded)
        
        return hidden.squeeze(0)


class MotionDynamicsEncoder(nn.Module):
    """
    Combined encoder for position, velocity, and acceleration.
    Captures full motion dynamics.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        velocity_dim: int = 64,
        accel_dim: int = 32
    ):
        super().__init__()
        
        self.velocity_encoder = VelocityEncoder(
            input_dim=2, 
            hidden_dim=velocity_dim
        )
        self.accel_encoder = AccelerationEncoder(
            input_dim=2,
            hidden_dim=accel_dim
        )
        
        # Combine velocity and acceleration
        self.combine = nn.Sequential(
            nn.Linear(velocity_dim + accel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Encode full motion dynamics.
        
        Args:
            velocity: [batch, seq_len, 2]
        
        Returns:
            dynamics_features: [batch, hidden_dim]
        """
        vel_features = self.velocity_encoder(velocity)
        accel_features = self.accel_encoder(velocity)
        
        combined = torch.cat([vel_features, accel_features], dim=-1)
        output = self.combine(combined)
        
        return output


class SpeedPredictor(nn.Module):
    """
    Auxiliary module to predict future speed from current motion state.
    Used for adaptive trajectory spacing.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Speed must be positive
        )
    
    def forward(self, motion_features: torch.Tensor) -> torch.Tensor:
        """
        Predict speed from motion features.
        
        Args:
            motion_features: [batch, input_dim]
        
        Returns:
            predicted_speed: [batch, 1]
        """
        return self.predictor(motion_features)
