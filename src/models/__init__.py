"""
Model architectures for adaptive pose-guided trajectory prediction.

Main models:
- SocialPoseModel: Complete pose-guided trajectory prediction (recommended)
- AdaptivePoseTrajectoryPredictor: Alternative full model implementation

Components:
- TrajectoryEncoder: LSTM-based trajectory encoding
- PoseTransformerEncoder: Transformer-based pose feature extraction
- VelocityEncoder: Speed and heading encoding
- SocialPoolingModule: Multi-agent interaction modeling
- AdaptiveTrajectoryDecoder: Trajectory generation with speed prediction

Usage:
    from src.models import SocialPoseModel, build_social_pose_model
    
    # From config
    model = build_social_pose_model(config)
    
    # Direct initialization
    model = SocialPoseModel(
        obs_len=8, pred_len=12,
        num_joints=22, pose_dim=3,  # JTA
        use_pose=True, use_velocity=True,
    )
    
    # Forward pass
    predictions, extras = model(obs_traj, obs_pose, obs_velocity)
    
    # Multi-modal sampling (Best-of-K)
    samples = model.sample(obs_traj, obs_pose, num_samples=20)
"""

# Trajectory encoder
from .trajectory_encoder import (
    TrajectoryEncoder,
    DisplacementEncoder,
    BidirectionalTrajectoryEncoder,
)

# Pose encoder
from .pose_encoder import (
    PositionalEncoding,
    PoseTransformerEncoder,
    JointWiseEncoder,
    SpatialTemporalPoseEncoder,
)

# Velocity encoder
from .velocity_encoder import (
    VelocityEncoder,
    AccelerationEncoder,
    MotionDynamicsEncoder,
    SpeedPredictor,
)

# Social pooling
from .social_pooling import (
    SocialPoolingModule,
    GridBasedPooling,
    AttentionPooling,
)

# Decoder
from .decoder import (
    AdaptiveTrajectoryDecoder,
    MultiModalDecoder,
    GoalConditionedDecoder,
)

# Full models
from .full_model import (
    AdaptivePoseTrajectoryPredictor,
    build_model,
    load_model,
)

# Social-Pose model (recommended)
from .social_pose import (
    SocialPoseModel,
    SocialPoseBaseline,
    SocialPoseWithPose,
    SocialPoseFull,
    build_social_pose_model,
    load_social_pose_model,
)

__all__ = [
    # Trajectory encoder
    'TrajectoryEncoder',
    'DisplacementEncoder',
    'BidirectionalTrajectoryEncoder',
    
    # Pose encoder
    'PositionalEncoding',
    'PoseTransformerEncoder',
    'JointWiseEncoder',
    'SpatialTemporalPoseEncoder',
    
    # Velocity encoder
    'VelocityEncoder',
    'AccelerationEncoder',
    'MotionDynamicsEncoder',
    'SpeedPredictor',
    
    # Social pooling
    'SocialPoolingModule',
    'GridBasedPooling',
    'AttentionPooling',
    
    # Decoder
    'AdaptiveTrajectoryDecoder',
    'MultiModalDecoder',
    'GoalConditionedDecoder',
    
    # Full models
    'AdaptivePoseTrajectoryPredictor',
    'build_model',
    'load_model',
    
    # Social-Pose model (recommended)
    'SocialPoseModel',
    'SocialPoseBaseline',
    'SocialPoseWithPose',
    'SocialPoseFull',
    'build_social_pose_model',
    'load_social_pose_model',
]
