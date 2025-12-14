"""
Data augmentation for trajectory prediction.
Includes rotation, scaling, noise, and speed augmentation.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrajectoryAugmentation:
    """
    Augmentation pipeline for trajectory and pose data.
    
    Key augmentations:
    - Random rotation: Rotate trajectory and pose
    - Random scaling: Scale trajectory and pose
    - Random noise: Add Gaussian noise
    - Speed augmentation: Temporal resampling (CRITICAL for adaptive spacing)
    - Horizontal flip: Mirror trajectory
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-180, 180),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.01,
        speed_range: Tuple[float, float] = (0.5, 2.0),
        flip_prob: float = 0.5,
        enable_rotation: bool = True,
        enable_scaling: bool = True,
        enable_noise: bool = True,
        enable_speed: bool = True,
        enable_flip: bool = True
    ):
        """
        Args:
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            noise_std: Standard deviation of Gaussian noise
            speed_range: Range of speed factors for temporal resampling
            flip_prob: Probability of horizontal flip
            enable_*: Enable/disable specific augmentations
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.speed_range = speed_range
        self.flip_prob = flip_prob
        
        self.enable_rotation = enable_rotation
        self.enable_scaling = enable_scaling
        self.enable_noise = enable_noise
        self.enable_speed = enable_speed
        self.enable_flip = enable_flip
    
    def __call__(
        self,
        obs_traj: np.ndarray,
        pred_traj: np.ndarray,
        obs_pose: Optional[np.ndarray] = None,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentation to trajectory and pose.
        
        Args:
            obs_traj: [obs_len, 2] observation trajectory
            pred_traj: [pred_len, 2] prediction trajectory
            obs_pose: [obs_len, J, D] observation pose (optional)
            prob: Probability of applying each augmentation
        
        Returns:
            Augmented (obs_traj, pred_traj, obs_pose)
        """
        # Combine trajectories for consistent augmentation
        full_traj = np.vstack([obs_traj, pred_traj])
        obs_len = len(obs_traj)
        
        # Random rotation
        if self.enable_rotation and np.random.random() < prob:
            full_traj, obs_pose = self.random_rotation(full_traj, obs_pose)
        
        # Random scaling
        if self.enable_scaling and np.random.random() < prob:
            full_traj, obs_pose = self.random_scaling(full_traj, obs_pose)
        
        # Random noise
        if self.enable_noise and np.random.random() < prob:
            full_traj = self.add_noise(full_traj)
            if obs_pose is not None:
                obs_pose = self.add_noise_pose(obs_pose)
        
        # Horizontal flip
        if self.enable_flip and np.random.random() < self.flip_prob:
            full_traj, obs_pose = self.horizontal_flip(full_traj, obs_pose)
        
        # Split back
        obs_traj = full_traj[:obs_len]
        pred_traj = full_traj[obs_len:]
        
        return obs_traj, pred_traj, obs_pose
    
    def random_rotation(
        self,
        trajectory: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random rotation to trajectory and pose.
        
        Args:
            trajectory: [T, 2] trajectory
            pose: [T, J, D] pose (optional)
        
        Returns:
            Rotated (trajectory, pose)
        """
        angle = np.random.uniform(*self.rotation_range) * np.pi / 180
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a],
                      [sin_a, cos_a]])
        
        # Rotate trajectory
        traj_rotated = trajectory @ R.T
        
        # Rotate pose
        pose_rotated = None
        if pose is not None:
            pose_rotated = pose.copy()
            # Only rotate XY (first 2 dimensions)
            pose_xy = pose[:, :, :2].reshape(-1, 2)
            pose_xy_rot = pose_xy @ R.T
            pose_rotated[:, :, :2] = pose_xy_rot.reshape(pose.shape[0], pose.shape[1], 2)
        
        return traj_rotated, pose_rotated
    
    def random_scaling(
        self,
        trajectory: np.ndarray,
        pose: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random scaling to trajectory and pose.
        
        Args:
            trajectory: [T, 2] trajectory
            pose: [T, J, D] pose (optional)
        
        Returns:
            Scaled (trajectory, pose)
        """
        scale = np.random.uniform(*self.scale_range)
        
        traj_scaled = trajectory * scale
        pose_scaled = pose * scale if pose is not None else None
        
        return traj_scaled, pose_scaled
    
    def add_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to trajectory.
        
        Args:
            trajectory: [T, 2] trajectory
        
        Returns:
            Noisy trajectory
        """
        noise = np.random.normal(0, self.noise_std, trajectory.shape)
        return trajectory + noise
    
    def add_noise_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to pose.
        
        Args:
            pose: [T, J, D] pose
        
        Returns:
            Noisy pose
        """
        noise = np.random.normal(0, self.noise_std * 0.5, pose.shape)
        return pose + noise
    
    def horizontal_flip(
        self,
        trajectory: np.ndarray,
        pose: Optional[np.ndarray] = None,
        dataset: str = 'coco'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply horizontal flip to trajectory and pose.
        
        Args:
            trajectory: [T, 2] trajectory
            pose: [T, J, D] pose (optional)
            dataset: 'jta' (22 joints) or 'coco' (17 joints)
        
        Returns:
            Flipped (trajectory, pose)
        """
        # Flip X coordinate
        traj_flipped = trajectory.copy()
        traj_flipped[:, 0] = -traj_flipped[:, 0]
        
        pose_flipped = None
        if pose is not None:
            pose_flipped = pose.copy()
            pose_flipped[:, :, 0] = -pose_flipped[:, :, 0]
            
            # Swap left/right keypoints based on dataset
            if dataset == 'jta' or pose.shape[1] == 22:
                # JTA format (22 joints)
                swap_pairs = [
                    (3, 7),   # clavicles
                    (4, 8),   # shoulders
                    (5, 9),   # elbows
                    (6, 10),  # wrists
                    (16, 19), # hips
                    (17, 20), # knees
                    (18, 21), # ankles
                ]
            else:
                # COCO format (17 joints)
                swap_pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            
            for left, right in swap_pairs:
                if left < pose.shape[1] and right < pose.shape[1]:
                    pose_flipped[:, [left, right], :] = pose_flipped[:, [right, left], :]
        
        return traj_flipped, pose_flipped
    
    def speed_augmentation(
        self,
        trajectory: np.ndarray,
        pose: Optional[np.ndarray] = None,
        obs_len: int = 8
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Temporal resampling to simulate different walking speeds.
        
        IMPORTANT: This is critical for learning adaptive spacing.
        - Slower speed factor < 1: simulates slower walking
        - Faster speed factor > 1: simulates faster walking/running
        
        Args:
            trajectory: [T, 2] trajectory
            pose: [T, J, D] pose (optional)
            obs_len: Observation length
        
        Returns:
            Resampled (trajectory, pose)
        """
        speed_factor = np.random.uniform(*self.speed_range)
        
        T = len(trajectory)
        new_T = int(T / speed_factor)
        
        if new_T < obs_len + 1:
            return trajectory, pose
        
        # Resample trajectory
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        traj_resampled = np.zeros((new_T, 2))
        traj_resampled[:, 0] = np.interp(new_indices, old_indices, trajectory[:, 0])
        traj_resampled[:, 1] = np.interp(new_indices, old_indices, trajectory[:, 1])
        
        # Resample pose
        pose_resampled = None
        if pose is not None:
            T_pose = len(pose)
            new_T_pose = int(T_pose / speed_factor)
            
            if new_T_pose >= obs_len:
                pose_resampled = np.zeros((new_T_pose, pose.shape[1], pose.shape[2]))
                old_indices_p = np.arange(T_pose)
                new_indices_p = np.linspace(0, T_pose - 1, new_T_pose)
                
                for j in range(pose.shape[1]):
                    for d in range(pose.shape[2]):
                        pose_resampled[:, j, d] = np.interp(
                            new_indices_p, old_indices_p, pose[:, j, d]
                        )
            else:
                pose_resampled = pose
        
        return traj_resampled, pose_resampled


class PoseAugmentation:
    """
    Pose-specific augmentation.
    """
    
    def __init__(
        self,
        joint_dropout_prob: float = 0.1,
        frame_dropout_prob: float = 0.05,
        jitter_std: float = 0.02
    ):
        """
        Args:
            joint_dropout_prob: Probability of dropping a joint
            frame_dropout_prob: Probability of dropping entire frame
            jitter_std: Standard deviation of joint jitter
        """
        self.joint_dropout_prob = joint_dropout_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.jitter_std = jitter_std
    
    def __call__(self, pose: np.ndarray) -> np.ndarray:
        """
        Apply pose augmentation.
        
        Args:
            pose: [T, J, D] pose array
        
        Returns:
            Augmented pose
        """
        pose_aug = pose.copy()
        
        # Joint dropout (simulate occlusion)
        if self.joint_dropout_prob > 0:
            mask = np.random.random(pose.shape[:2]) > self.joint_dropout_prob
            pose_aug = pose_aug * mask[:, :, np.newaxis]
        
        # Frame dropout
        if self.frame_dropout_prob > 0:
            frame_mask = np.random.random(pose.shape[0]) > self.frame_dropout_prob
            # For dropped frames, use previous frame
            for t in range(1, len(pose_aug)):
                if not frame_mask[t]:
                    pose_aug[t] = pose_aug[t - 1]
        
        # Joint jitter
        if self.jitter_std > 0:
            jitter = np.random.normal(0, self.jitter_std, pose_aug.shape)
            pose_aug = pose_aug + jitter
        
        return pose_aug


def mixup_trajectories(
    traj1: np.ndarray,
    traj2: np.ndarray,
    alpha: float = 0.2
) -> np.ndarray:
    """
    Mixup augmentation for trajectories.
    
    Args:
        traj1: [T, 2] first trajectory
        traj2: [T, 2] second trajectory
        alpha: Mixup coefficient
    
    Returns:
        Mixed trajectory
    """
    lam = np.random.beta(alpha, alpha)
    return lam * traj1 + (1 - lam) * traj2


def cutout_trajectory(
    trajectory: np.ndarray,
    num_frames: int = 2
) -> np.ndarray:
    """
    Cutout augmentation: zero out random frames.
    
    Args:
        trajectory: [T, 2] trajectory
        num_frames: Number of frames to cut out
    
    Returns:
        Trajectory with cut out frames
    """
    traj_aug = trajectory.copy()
    T = len(trajectory)
    
    if num_frames >= T:
        return traj_aug
    
    # Random start position
    start = np.random.randint(0, T - num_frames)
    
    # Zero out frames
    traj_aug[start:start + num_frames] = 0
    
    return traj_aug
