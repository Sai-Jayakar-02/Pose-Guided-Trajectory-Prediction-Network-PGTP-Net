"""
Complete Adaptive Pose-Trajectory Prediction Model.
Combines trajectory encoder, pose encoder, velocity encoder, social pooling, and decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .trajectory_encoder import TrajectoryEncoder
from .pose_encoder import PoseTransformerEncoder
from .velocity_encoder import VelocityEncoder
from .social_pooling import SocialPoolingModule
from .decoder import AdaptiveTrajectoryDecoder


class AdaptivePoseTrajectoryPredictor(nn.Module):
    """
    Complete model for adaptive pose-guided trajectory prediction.
    
    Architecture:
    1. Trajectory Encoder (LSTM): Encodes past positions
    2. Pose Encoder (Transformer): Extracts body intent features
    3. Velocity Encoder (LSTM): Encodes speed and heading
    4. Social Pooling: Models multi-agent interactions
    5. Feature Fusion: Concatenates all features
    6. Trajectory Decoder: Generates future predictions with speed
    
    Key Innovation:
    - Uses pose to detect intent BEFORE trajectory changes
    - Predicts speed at each timestep for adaptive spacing
    - Supports multiple pedestrians with social pooling
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Store config
        self.config = config
        
        # Extract parameters
        self.obs_len = config.get('obs_len', 8)
        self.pred_len = config.get('pred_len', 12)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_joints = config.get('num_joints', 17)
        self.pose_dim = config.get('pose_dim', 3)
        
        # Flags for optional components
        self.use_pose = config.get('use_pose', True)
        self.use_velocity = config.get('use_velocity', True)
        self.use_social = config.get('use_social', False)
        
        # Trajectory Encoder
        traj_config = config.get('trajectory_encoder', {})
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=traj_config.get('input_dim', 2),
            embedding_dim=traj_config.get('embedding_dim', 64),
            hidden_dim=self.hidden_dim,
            num_layers=traj_config.get('num_layers', 1),
            dropout=traj_config.get('dropout', 0.0)
        )
        
        # Pose Encoder (optional)
        if self.use_pose:
            pose_config = config.get('pose_encoder', {})
            self.pose_encoder = PoseTransformerEncoder(
                num_joints=self.num_joints,
                input_dim=self.pose_dim,
                hidden_dim=self.hidden_dim,
                num_heads=pose_config.get('num_heads', 8),
                num_layers=pose_config.get('num_layers', 2),
                dropout=pose_config.get('dropout', 0.1)
            )
        else:
            self.pose_encoder = None
        
        # Velocity Encoder (optional)
        if self.use_velocity:
            vel_config = config.get('velocity_encoder', {})
            self.velocity_encoder = VelocityEncoder(
                input_dim=vel_config.get('input_dim', 2),
                hidden_dim=vel_config.get('hidden_dim', 64)
            )
            velocity_dim = vel_config.get('hidden_dim', 64)
        else:
            self.velocity_encoder = None
            velocity_dim = 0
        
        # Social Pooling (optional)
        if self.use_social:
            social_config = config.get('social_pooling', {})
            self.social_pooling = SocialPoolingModule(
                hidden_dim=self.hidden_dim,
                pooling_dim=social_config.get('pooling_dim', 256)
            )
            social_dim = social_config.get('pooling_dim', 256)
        else:
            self.social_pooling = None
            social_dim = 0
        
        # Feature Fusion
        # Concatenate: trajectory + pose + velocity
        fusion_input_dim = self.hidden_dim  # trajectory
        if self.use_pose:
            fusion_input_dim += self.hidden_dim  # pose
        if self.use_velocity:
            fusion_input_dim += velocity_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Trajectory Decoder
        decoder_config = config.get('decoder', {})
        self.decoder = AdaptiveTrajectoryDecoder(
            hidden_dim=self.hidden_dim,
            embedding_dim=decoder_config.get('embedding_dim', 64),
            output_dim=decoder_config.get('output_dim', 2),
            pred_len=self.pred_len,
            predict_speed=decoder_config.get('predict_speed', True),
            predict_uncertainty=decoder_config.get('predict_uncertainty', True)
        )
    
    def forward(
        self,
        obs_traj: torch.Tensor,
        obs_pose: Optional[torch.Tensor] = None,
        obs_velocity: Optional[torch.Tensor] = None,
        neighbors_traj: Optional[torch.Tensor] = None,
        neighbors_pose: Optional[torch.Tensor] = None,
        gt_trajectory: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass.
        
        Args:
            obs_traj: [batch, obs_len, 2] - observed trajectory
            obs_pose: [batch, obs_len, num_joints, pose_dim] - observed pose
            obs_velocity: [batch, obs_len-1, 2] - observed velocities
            neighbors_traj: [batch, num_neighbors, obs_len, 2] - neighbor trajectories
            neighbors_pose: [batch, num_neighbors, obs_len, num_joints, pose_dim]
            gt_trajectory: [batch, pred_len, 2] - ground truth for teacher forcing
            teacher_forcing_ratio: Probability of using ground truth
        
        Returns:
            predictions: [batch, pred_len, 2]
            speeds: [batch, pred_len, 1] (if predicting speed)
            uncertainties: [batch, pred_len, 2] (if predicting uncertainty)
        """
        batch_size = obs_traj.size(0)
        
        # 1. Encode trajectory
        traj_hidden, traj_cell = self.trajectory_encoder(obs_traj)
        
        # 2. Encode pose (if available)
        if self.use_pose and obs_pose is not None:
            pose_features = self.pose_encoder(obs_pose)
        else:
            pose_features = None
        
        # 3. Encode velocity (if available)
        if self.use_velocity and obs_velocity is not None:
            vel_features = self.velocity_encoder(obs_velocity)
        else:
            vel_features = None
        
        # 4. Fuse features (concatenation)
        features_to_fuse = [traj_hidden]
        
        if pose_features is not None:
            features_to_fuse.append(pose_features)
        
        if vel_features is not None:
            features_to_fuse.append(vel_features)
        
        fused = torch.cat(features_to_fuse, dim=-1)
        fused = self.fusion(fused)
        
        # 5. Social pooling (if enabled and neighbors provided)
        social_context = None
        if self.use_social and neighbors_traj is not None:
            # Encode all neighbor trajectories
            num_neighbors = neighbors_traj.size(1)
            
            if num_neighbors > 0:
                # Reshape for batch processing
                neighbors_flat = neighbors_traj.view(-1, self.obs_len, 2)
                neighbors_hidden, _ = self.trajectory_encoder(neighbors_flat)
                neighbors_hidden = neighbors_hidden.view(batch_size, num_neighbors, -1)
                
                # Get positions at last observation
                all_positions = torch.cat([
                    obs_traj[:, -1:, :],  # Target position
                    neighbors_traj[:, :, -1, :]  # Neighbor positions
                ], dim=1)
                
                all_hidden = torch.cat([
                    fused.unsqueeze(1),
                    neighbors_hidden
                ], dim=1)
                
                # Pool
                pooled = self.social_pooling(all_hidden, all_positions)
                social_context = pooled[:, 0, :]  # Target's social context
        
        # 6. Decode future trajectory
        last_pos = obs_traj[:, -1, :]
        
        predictions, speeds, uncertainties = self.decoder(
            encoder_hidden=fused,
            encoder_cell=traj_cell,
            last_position=last_pos,
            social_context=social_context,
            teacher_forcing_ratio=teacher_forcing_ratio,
            gt_trajectory=gt_trajectory
        )
        
        return predictions, speeds, uncertainties
    
    def predict(
        self,
        obs_traj: torch.Tensor,
        obs_pose: Optional[torch.Tensor] = None,
        obs_velocity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-mode prediction (no teacher forcing).
        
        Args:
            obs_traj: [batch, obs_len, 2]
            obs_pose: [batch, obs_len, num_joints, pose_dim]
            obs_velocity: [batch, obs_len-1, 2]
        
        Returns:
            Dictionary with predictions, speeds, uncertainties
        """
        self.eval()
        
        with torch.no_grad():
            predictions, speeds, uncertainties = self.forward(
                obs_traj=obs_traj,
                obs_pose=obs_pose,
                obs_velocity=obs_velocity,
                teacher_forcing_ratio=0.0
            )
        
        result = {'predictions': predictions}
        
        if speeds is not None:
            result['speeds'] = speeds
        
        if uncertainties is not None:
            result['uncertainties'] = uncertainties
        
        return result
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: Dict) -> AdaptivePoseTrajectoryPredictor:
    """
    Factory function to build model from config.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    model = AdaptivePoseTrajectoryPredictor(config)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


def load_model(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[AdaptivePoseTrajectoryPredictor, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = AdaptivePoseTrajectoryPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config
