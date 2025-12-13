"""
Data preprocessing utilities for trajectory prediction.
Includes normalization, velocity computation, and sequence creation.

Supports:
- JTA dataset (22 joints, 3D coordinates)
- ETH/UCY dataset (trajectory only)
- Custom datasets
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# JTA Constants (from official documentation)
# =============================================================================
JTA_NUM_JOINTS = 22
JTA_PELVIS_IDX = 13  # spine2 - use as center
JTA_NECK_IDX = 2

# Joint indices for trajectory prediction
JTA_KEY_JOINTS = {
    'pelvis': 13,       # spine2 - center point
    'neck': 2,          # neck
    'shoulders': [4, 8],  # right/left shoulder
    'hips': [16, 19],     # right/left hip
    'knees': [17, 20],    # right/left knee
    'ankles': [18, 21],   # right/left ankle
    'wrists': [6, 10],    # right/left wrist
    'elbows': [5, 9],     # right/left elbow
}

# Left-right swap pairs for flip augmentation
JTA_FLIP_PAIRS = [
    (3, 7),   # clavicles
    (4, 8),   # shoulders
    (5, 9),   # elbows
    (6, 10),  # wrists
    (16, 19), # hips
    (17, 20), # knees
    (18, 21), # ankles
]

# COCO format (17 joints) for pose extraction
COCO_NUM_JOINTS = 17
COCO_PELVIS_IDX = 0  # Usually nose, but we use hip midpoint
COCO_KEY_JOINTS = {
    'nose': 0,
    'shoulders': [5, 6],
    'elbows': [7, 8],
    'wrists': [9, 10],
    'hips': [11, 12],
    'knees': [13, 14],
    'ankles': [15, 16],
}
COCO_FLIP_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def normalize_trajectory(
    trajectory: np.ndarray,
    mode: str = 'last_obs',
    obs_len: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize trajectory to a reference point.
    
    Args:
        trajectory: [T, 2] array of positions
        mode: Normalization mode
            - 'last_obs': Origin at last observation frame
            - 'first': Origin at first frame
            - 'mean': Origin at mean position
        obs_len: Observation length (for 'last_obs' mode)
    
    Returns:
        normalized: Normalized trajectory
        origin: The reference point used for normalization
    """
    if mode == 'last_obs':
        origin = trajectory[obs_len - 1].copy()
    elif mode == 'first':
        origin = trajectory[0].copy()
    elif mode == 'mean':
        origin = trajectory.mean(axis=0)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    
    normalized = trajectory - origin
    return normalized, origin


def denormalize_trajectory(
    trajectory: np.ndarray,
    origin: np.ndarray
) -> np.ndarray:
    """
    Denormalize trajectory back to world coordinates.
    
    Args:
        trajectory: Normalized trajectory [T, 2]
        origin: Reference point used for normalization
    
    Returns:
        Denormalized trajectory in world coordinates
    """
    return trajectory + origin


def normalize_pose(
    pose: np.ndarray,
    pelvis_idx: int = 13,  # JTA: spine2 (13), COCO: hip midpoint
    scale_by_torso: bool = False,
    neck_idx: int = 2,     # JTA: neck (2), COCO: varies
    dataset: str = 'jta'   # 'jta' or 'coco'
) -> np.ndarray:
    """
    Normalize pose to be pelvis-centered.
    
    For JTA: pelvis_idx=13 (spine2), neck_idx=2
    For COCO: compute hip midpoint as pelvis
    
    Args:
        pose: [T, J, D] array of keypoints (T frames, J joints, D dimensions)
        pelvis_idx: Index of the pelvis/hip joint (13 for JTA spine2)
        scale_by_torso: Whether to scale by torso length for size invariance
        neck_idx: Index of neck joint for torso length computation
        dataset: 'jta' or 'coco' to use appropriate defaults
    
    Returns:
        Normalized pose
    """
    if pose.ndim != 3:
        raise ValueError(f"Expected 3D pose array [T, J, D], got shape {pose.shape}")
    
    T, J, D = pose.shape
    
    # For COCO, compute pelvis as midpoint of hips
    if dataset == 'coco' and J == COCO_NUM_JOINTS:
        # Hip indices in COCO are 11 and 12
        pelvis = (pose[:, 11:12, :] + pose[:, 12:13, :]) / 2
    else:
        # Extract pelvis position for each frame
        pelvis = pose[:, pelvis_idx:pelvis_idx+1, :]  # [T, 1, D]
    
    # Subtract pelvis from all joints
    normalized = pose - pelvis
    
    if scale_by_torso:
        # Compute torso length (pelvis to neck)
        torso_length = compute_torso_length(pose, pelvis_idx, neck_idx, dataset)
        torso_length = np.maximum(torso_length, 0.1)  # Avoid division by zero
        normalized = normalized / torso_length[:, np.newaxis, np.newaxis]
    
    return normalized


def compute_torso_length(
    pose: np.ndarray,
    pelvis_idx: int = 13,   # JTA: spine2 (13)
    neck_idx: int = 2,      # JTA: neck (2)
    dataset: str = 'jta'
) -> np.ndarray:
    """
    Compute torso length from pelvis to neck.
    
    For JTA: pelvis=spine2(13), neck=neck(2)
    For COCO: pelvis=hip_midpoint, neck varies
    
    Args:
        pose: [T, J, D] pose array
        pelvis_idx: Pelvis joint index
        neck_idx: Neck joint index
        dataset: 'jta' or 'coco'
    
    Returns:
        [T,] array of torso lengths per frame
    """
    T, J, D = pose.shape
    
    if dataset == 'coco' and J == COCO_NUM_JOINTS:
        # COCO: use hip midpoint as pelvis
        pelvis = (pose[:, 11, :] + pose[:, 12, :]) / 2
        # Use shoulder midpoint as neck proxy
        neck = (pose[:, 5, :] + pose[:, 6, :]) / 2
    else:
        pelvis = pose[:, pelvis_idx, :]
        neck = pose[:, neck_idx, :]
    
    return np.linalg.norm(neck - pelvis, axis=1)


def compute_velocity(
    positions: np.ndarray,
    dt: float = 0.4
) -> np.ndarray:
    """
    Compute velocity from positions.
    
    Args:
        positions: [T, 2] array of positions
        dt: Time step between frames (0.4s for 2.5 FPS)
    
    Returns:
        velocities: [T-1, 2] array of velocity vectors
    """
    velocities = np.diff(positions, axis=0) / dt
    return velocities


def compute_speed(velocities: np.ndarray) -> np.ndarray:
    """
    Compute speed (magnitude) from velocities.
    
    Args:
        velocities: [T, 2] array of velocity vectors
    
    Returns:
        speeds: [T, 1] array of speed values
    """
    return np.linalg.norm(velocities, axis=1, keepdims=True)


def compute_acceleration(
    velocities: np.ndarray,
    dt: float = 0.4
) -> np.ndarray:
    """
    Compute acceleration from velocities.
    
    Args:
        velocities: [T, 2] array of velocities
        dt: Time step
    
    Returns:
        accelerations: [T-1, 2] array
    """
    return np.diff(velocities, axis=0) / dt


def compute_heading(velocities: np.ndarray) -> np.ndarray:
    """
    Compute heading angle from velocities.
    
    Args:
        velocities: [T, 2] array of velocity vectors
    
    Returns:
        headings: [T,] array of angles in radians
    """
    return np.arctan2(velocities[:, 1], velocities[:, 0])


def compute_heading_change(headings: np.ndarray) -> np.ndarray:
    """
    Compute change in heading angle.
    
    Args:
        headings: [T,] array of angles in radians
    
    Returns:
        heading_changes: [T-1,] array of angle changes
    """
    changes = np.diff(headings)
    # Normalize to [-pi, pi]
    changes = (changes + np.pi) % (2 * np.pi) - np.pi
    return changes


def create_sequences(
    trajectories: List[np.ndarray],
    poses: Optional[List[np.ndarray]] = None,
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
    min_length: Optional[int] = None
) -> List[Dict]:
    """
    Create training sequences from trajectory data using sliding window.
    
    Args:
        trajectories: List of [T, 2] trajectory arrays
        poses: Optional list of [T, J, D] pose arrays
        obs_len: Observation length
        pred_len: Prediction length
        skip: Step size for sliding window
        min_length: Minimum trajectory length to consider
    
    Returns:
        List of sequence dictionaries
    """
    seq_len = obs_len + pred_len
    min_length = min_length or seq_len
    
    sequences = []
    
    for i, traj in enumerate(trajectories):
        if len(traj) < min_length:
            continue
        
        pose = poses[i] if poses is not None else None
        
        # Sliding window
        for start in range(0, len(traj) - seq_len + 1, skip):
            end = start + seq_len
            
            seq_traj = traj[start:end]
            seq_pose = pose[start:end] if pose is not None else None
            
            seq_data = {
                'trajectory': seq_traj,
                'has_pose': pose is not None
            }
            
            if seq_pose is not None:
                seq_data['pose'] = seq_pose
            
            sequences.append(seq_data)
    
    logger.info(f"Created {len(sequences)} sequences from {len(trajectories)} trajectories")
    return sequences


def world_to_pixel(
    world_coords: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    Convert world coordinates to pixel coordinates using homography.
    
    Args:
        world_coords: [N, 2] world coordinates
        homography: [3, 3] homography matrix
    
    Returns:
        pixel_coords: [N, 2] pixel coordinates
    """
    N = world_coords.shape[0]
    
    # Add homogeneous coordinate
    ones = np.ones((N, 1))
    world_homo = np.hstack([world_coords, ones])
    
    # Apply homography
    pixel_homo = world_homo @ homography.T
    
    # Normalize
    pixel_coords = pixel_homo[:, :2] / pixel_homo[:, 2:3]
    
    return pixel_coords


def pixel_to_world(
    pixel_coords: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    Convert pixel coordinates to world coordinates using inverse homography.
    
    Args:
        pixel_coords: [N, 2] pixel coordinates
        homography: [3, 3] homography matrix (world to pixel)
    
    Returns:
        world_coords: [N, 2] world coordinates
    """
    H_inv = np.linalg.inv(homography)
    return world_to_pixel(pixel_coords, H_inv)


def interpolate_trajectory(
    trajectory: np.ndarray,
    target_len: int,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate trajectory to target length.
    
    Args:
        trajectory: [T, 2] original trajectory
        target_len: Desired length
        method: Interpolation method ('linear', 'cubic')
    
    Returns:
        Interpolated trajectory [target_len, 2]
    """
    from scipy import interpolate
    
    T = len(trajectory)
    
    if T == target_len:
        return trajectory
    
    # Create interpolation functions
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, target_len)
    
    if method == 'linear':
        interp_x = interpolate.interp1d(t_orig, trajectory[:, 0])
        interp_y = interpolate.interp1d(t_orig, trajectory[:, 1])
    elif method == 'cubic':
        interp_x = interpolate.interp1d(t_orig, trajectory[:, 0], kind='cubic')
        interp_y = interpolate.interp1d(t_orig, trajectory[:, 1], kind='cubic')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    new_traj = np.stack([interp_x(t_new), interp_y(t_new)], axis=1)
    
    return new_traj


def resample_by_distance(
    trajectory: np.ndarray,
    spacing: float = 0.5
) -> np.ndarray:
    """
    Resample trajectory to have uniform distance between points.
    This is used for adaptive spacing visualization.
    
    Args:
        trajectory: [T, 2] trajectory
        spacing: Desired distance between points (meters)
    
    Returns:
        Resampled trajectory with uniform spacing
    """
    # Compute cumulative distance
    distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    total_distance = cumulative[-1]
    
    if total_distance < spacing:
        return trajectory
    
    # Generate evenly spaced distance values
    num_points = int(total_distance / spacing) + 1
    target_distances = np.linspace(0, total_distance, num_points)
    
    # Interpolate positions at target distances
    resampled = np.zeros((num_points, 2))
    for i, d in enumerate(target_distances):
        # Find segment containing this distance
        idx = np.searchsorted(cumulative, d) - 1
        idx = max(0, min(idx, len(trajectory) - 2))
        
        # Interpolate within segment
        seg_start = cumulative[idx]
        seg_end = cumulative[idx + 1]
        seg_len = seg_end - seg_start
        
        if seg_len > 0:
            alpha = (d - seg_start) / seg_len
        else:
            alpha = 0
        
        resampled[i] = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
    
    return resampled


def compute_social_features(
    target_traj: np.ndarray,
    neighbor_trajs: List[np.ndarray],
    obs_len: int = 8
) -> Dict[str, np.ndarray]:
    """
    Compute social interaction features between target and neighbors.
    
    Args:
        target_traj: [T, 2] target trajectory
        neighbor_trajs: List of [T, 2] neighbor trajectories
        obs_len: Observation length
    
    Returns:
        Dictionary of social features
    """
    if len(neighbor_trajs) == 0:
        return {
            'relative_positions': np.zeros((obs_len, 0, 2)),
            'distances': np.zeros((obs_len, 0)),
            'num_neighbors': 0
        }
    
    # Stack neighbor trajectories
    neighbors = np.stack(neighbor_trajs, axis=1)  # [T, N, 2]
    
    # Compute relative positions
    target_obs = target_traj[:obs_len, np.newaxis, :]  # [obs_len, 1, 2]
    neighbors_obs = neighbors[:obs_len]  # [obs_len, N, 2]
    
    relative_positions = neighbors_obs - target_obs  # [obs_len, N, 2]
    
    # Compute distances
    distances = np.linalg.norm(relative_positions, axis=2)  # [obs_len, N]
    
    return {
        'relative_positions': relative_positions,
        'distances': distances,
        'num_neighbors': len(neighbor_trajs)
    }


def filter_static_pedestrians(
    trajectory: np.ndarray,
    threshold: float = 0.1
) -> bool:
    """
    Check if a pedestrian is moving (not static).
    
    Args:
        trajectory: [T, 2] trajectory
        threshold: Minimum displacement to be considered moving
    
    Returns:
        True if pedestrian is moving, False if static
    """
    total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    return total_displacement > threshold


def smooth_trajectory(
    trajectory: np.ndarray,
    window_size: int = 3,
    method: str = 'moving_average'
) -> np.ndarray:
    """
    Smooth a trajectory to reduce noise.
    
    Args:
        trajectory: [T, 2] trajectory
        window_size: Smoothing window size
        method: 'moving_average' or 'gaussian'
    
    Returns:
        Smoothed trajectory
    """
    if method == 'moving_average':
        kernel = np.ones(window_size) / window_size
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(trajectory, sigma=window_size/3, axis=0)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    smoothed = np.zeros_like(trajectory)
    for dim in range(trajectory.shape[1]):
        smoothed[:, dim] = np.convolve(trajectory[:, dim], kernel, mode='same')
    
    return smoothed
