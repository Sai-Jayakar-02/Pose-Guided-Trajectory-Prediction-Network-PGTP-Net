"""
Dataset classes for trajectory prediction.
Supports ETH/UCY (TrajPRed format) and JTA (native pose annotations).

JTA Format (each JSON is a matrix - list of lists):
    [frame_number, person_id, joint_type, x2D, y2D, x3D, y3D, z3D, occluded, self_occluded]

ETH/UCY Format (txt files):
    frame_id    pedestrian_id    x    y
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging

from .preprocessing import (
    normalize_trajectory,
    normalize_pose,
    compute_velocity,
    compute_speed,
)
from .augmentation import TrajectoryAugmentation

logger = logging.getLogger(__name__)


# =============================================================================
# JTA Constants (from official documentation)
# =============================================================================
JTA_JOINTS = {
    0: "head_top",
    1: "head_center", 
    2: "neck",
    3: "right_clavicle",
    4: "right_shoulder",
    5: "right_elbow",
    6: "right_wrist",
    7: "left_clavicle",
    8: "left_shoulder",
    9: "left_elbow",
    10: "left_wrist",
    11: "spine0",
    12: "spine1",
    13: "spine2",       # Use as pelvis/center
    14: "spine3",
    15: "spine4",
    16: "right_hip",
    17: "right_knee",
    18: "right_ankle",
    19: "left_hip",
    20: "left_knee",
    21: "left_ankle",
}

JTA_NUM_JOINTS = 22
JTA_PELVIS_IDX = 13  # spine2
JTA_NECK_IDX = 2

# Key joints for trajectory prediction
JTA_KEY_JOINTS = {
    'pelvis': 13,
    'shoulders': [4, 8],
    'hips': [16, 19],
    'knees': [17, 20],
    'ankles': [18, 21],
    'wrists': [6, 10],
}

# Left-right swap pairs for flipping augmentation
JTA_FLIP_PAIRS = [
    (3, 7),   # right/left clavicle
    (4, 8),   # right/left shoulder
    (5, 9),   # right/left elbow
    (6, 10),  # right/left wrist
    (16, 19), # right/left hip
    (17, 20), # right/left knee
    (18, 21), # right/left ankle
]

# JTA Camera intrinsics
JTA_CAMERA = {
    'fx': 1158,
    'fy': 1158,
    'cx': 960,
    'cy': 540,
    'resolution': (1920, 1080),
}


class TrajectoryDataset(Dataset):
    """
    Base dataset class for trajectory prediction.
    Handles trajectory-only data (no pose).
    """
    
    def __init__(
        self,
        trajectories: np.ndarray,
        obs_len: int = 8,
        pred_len: int = 12,
        transform=None,
        normalize: bool = True,
    ):
        """
        Args:
            trajectories: [N, T, 2] array of trajectories
            obs_len: Number of observation frames
            pred_len: Number of prediction frames
            transform: Optional augmentation transform
            normalize: Whether to normalize trajectories
        """
        super().__init__()
        
        self.trajectories = trajectories
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.transform = transform
        self.normalize = normalize
        
        logger.info(f"TrajectoryDataset: {len(self.trajectories)} sequences")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx].copy()
        
        # Split into observation and prediction
        obs_traj = trajectory[:self.obs_len]
        pred_traj = trajectory[self.obs_len:self.obs_len + self.pred_len]
        
        # Apply transform
        if self.transform is not None:
            obs_traj, pred_traj, _ = self.transform(obs_traj, pred_traj)
        
        # Normalize (origin at last observation)
        origin = obs_traj[-1].copy()
        if self.normalize:
            obs_traj = obs_traj - origin
            pred_traj = pred_traj - origin
        
        # Compute velocity
        full_traj = np.vstack([obs_traj, pred_traj[:1]])
        velocities = np.diff(full_traj, axis=0)
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        
        return {
            'obs_traj': torch.FloatTensor(obs_traj),
            'pred_traj': torch.FloatTensor(pred_traj),
            'obs_velocity': torch.FloatTensor(velocities),
            'obs_speed': torch.FloatTensor(speeds),
            'origin': torch.FloatTensor(origin),
        }


class JTADataset(Dataset):
    """
    JTA (Joint Track Auto) dataset with native 3D pose annotations.
    
    JTA annotation format (each row in JSON matrix):
        [frame_number, person_id, joint_type, x2D, y2D, x3D, y3D, z3D, occluded, self_occluded]
    
    Note: Frames are 1-indexed in JTA.
    """
    
    # Column indices in annotation matrix
    COL_FRAME = 0
    COL_PERSON = 1
    COL_JOINT = 2
    COL_X2D = 3
    COL_Y2D = 4
    COL_X3D = 5
    COL_Y3D = 6
    COL_Z3D = 7
    COL_OCCLUDED = 8
    COL_SELF_OCCLUDED = 9
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_trajectory_len: int = 20,
        use_3d: bool = True,
        downsample_factor: int = 12,  # 30fps -> 2.5fps
        max_sequences_per_video: int = None,
        augment: bool = False,
        occlusion_threshold: float = 0.5,
    ):
        """
        Args:
            data_dir: Root directory of JTA dataset (contains annotations/, videos/)
            split: 'train', 'val', or 'test'
            obs_len: Observation length in frames (at target fps)
            pred_len: Prediction length in frames (at target fps)
            skip: Sliding window stride
            min_trajectory_len: Minimum frames a person must appear
            use_3d: Use 3D coordinates (meters) vs 2D (pixels)
            downsample_factor: Factor to downsample from 30fps to 2.5fps
            max_sequences_per_video: Limit sequences per video (for debugging)
            augment: Whether to apply data augmentation
            occlusion_threshold: Max fraction of occluded joints allowed
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_trajectory_len = min_trajectory_len
        self.use_3d = use_3d
        self.downsample_factor = downsample_factor
        self.max_sequences_per_video = max_sequences_per_video
        self.occlusion_threshold = occlusion_threshold
        
        # Augmentation
        self.augment = augment
        self.augmentor = TrajectoryAugmentation(
            enable_speed=True,  # Critical for adaptive spacing
        ) if augment else None
        
        # Load sequences
        self.sequences = []
        self._load_data()
        
        logger.info(f"JTADataset ({split}): Loaded {len(self.sequences)} sequences")
    
    def _load_data(self):
        """Load and process all annotation files."""
        ann_dir = os.path.join(self.data_dir, 'annotations', self.split)
        
        if not os.path.exists(ann_dir):
            raise FileNotFoundError(f"JTA annotations not found at {ann_dir}")
        
        json_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
        logger.info(f"Found {len(json_files)} annotation files in {ann_dir}")
        
        for jf in json_files:
            filepath = os.path.join(ann_dir, jf)
            sequences = self._process_annotation_file(filepath)
            
            if self.max_sequences_per_video and len(sequences) > self.max_sequences_per_video:
                sequences = sequences[:self.max_sequences_per_video]
            
            self.sequences.extend(sequences)
    
    def _process_annotation_file(self, filepath: str) -> List[Dict]:
        """
        Process a single JTA annotation file.
        
        Returns list of sequence dicts with keys:
            - trajectory: [T, 2] positions (using pelvis joint)
            - pose: [T, 22, 3] all joint positions
            - person_id: person identifier
            - video_name: source video name
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # data is a list of lists (matrix)
        # Group by person_id and frame
        person_frames = defaultdict(lambda: defaultdict(dict))
        
        for row in data:
            frame = int(row[self.COL_FRAME])
            person_id = int(row[self.COL_PERSON])
            joint_type = int(row[self.COL_JOINT])
            
            if self.use_3d:
                position = [row[self.COL_X3D], row[self.COL_Y3D], row[self.COL_Z3D]]
            else:
                position = [row[self.COL_X2D], row[self.COL_Y2D], 0]
            
            occluded = row[self.COL_OCCLUDED] if len(row) > self.COL_OCCLUDED else 0
            
            person_frames[person_id][frame][joint_type] = {
                'position': position,
                'occluded': occluded,
            }
        
        # Create sequences for each person
        sequences = []
        video_name = os.path.basename(filepath).replace('.json', '')
        
        for person_id, frames_dict in person_frames.items():
            person_seqs = self._create_person_sequences(
                frames_dict, person_id, video_name
            )
            sequences.extend(person_seqs)
        
        return sequences
    
    def _create_person_sequences(
        self,
        frames_dict: Dict,
        person_id: int,
        video_name: str
    ) -> List[Dict]:
        """Create sequences for a single person using sliding window."""
        frames = sorted(frames_dict.keys())
        
        if len(frames) < self.min_trajectory_len:
            return []
        
        # Downsample frames (30fps -> 2.5fps)
        downsampled_frames = frames[::self.downsample_factor]
        
        if len(downsampled_frames) < self.seq_len:
            return []
        
        sequences = []
        
        # Sliding window
        for start_idx in range(0, len(downsampled_frames) - self.seq_len + 1, self.skip):
            seq_frames = downsampled_frames[start_idx:start_idx + self.seq_len]
            
            # Extract trajectory and pose for this sequence
            trajectory = []
            poses = []
            valid = True
            
            for frame in seq_frames:
                frame_data = frames_dict[frame]
                
                # Check if pelvis joint exists
                if JTA_PELVIS_IDX not in frame_data:
                    valid = False
                    break
                
                # Trajectory: use pelvis (spine2) as position
                pelvis_data = frame_data[JTA_PELVIS_IDX]
                if self.use_3d:
                    # Use X and Z for ground plane (Y is up in JTA)
                    traj_pos = [pelvis_data['position'][0], pelvis_data['position'][2]]
                else:
                    traj_pos = pelvis_data['position'][:2]
                trajectory.append(traj_pos)
                
                # Pose: all 22 joints
                pose = np.zeros((JTA_NUM_JOINTS, 3))
                occluded_count = 0
                
                for joint_idx in range(JTA_NUM_JOINTS):
                    if joint_idx in frame_data:
                        joint_data = frame_data[joint_idx]
                        pose[joint_idx] = joint_data['position']
                        if joint_data['occluded']:
                            occluded_count += 1
                
                # Check occlusion threshold
                if occluded_count / JTA_NUM_JOINTS > self.occlusion_threshold:
                    valid = False
                    break
                
                poses.append(pose)
            
            if not valid:
                continue
            
            sequences.append({
                'trajectory': np.array(trajectory),
                'pose': np.array(poses),
                'person_id': person_id,
                'video_name': video_name,
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        trajectory = seq['trajectory'].copy()  # [T, 2]
        pose = seq['pose'].copy()  # [T, 22, 3]
        
        # Split into observation and prediction
        obs_traj = trajectory[:self.obs_len]
        pred_traj = trajectory[self.obs_len:]
        obs_pose = pose[:self.obs_len]
        
        # Apply augmentation
        if self.augment and self.augmentor is not None:
            obs_traj, pred_traj, obs_pose = self.augmentor(
                obs_traj, pred_traj, obs_pose
            )
        
        # Normalize trajectory (origin at last observation)
        origin = obs_traj[-1].copy()
        obs_traj_norm = obs_traj - origin
        pred_traj_norm = pred_traj - origin
        
        # Normalize pose (pelvis-centered)
        obs_pose_norm = normalize_pose(obs_pose, pelvis_idx=JTA_PELVIS_IDX)
        
        # Compute velocity
        velocities = compute_velocity(np.vstack([obs_traj_norm, pred_traj_norm[:1]]))
        speeds = compute_speed(velocities)
        
        return {
            'obs_traj': torch.FloatTensor(obs_traj_norm),
            'pred_traj': torch.FloatTensor(pred_traj_norm),
            'obs_pose': torch.FloatTensor(obs_pose_norm),
            'obs_velocity': torch.FloatTensor(velocities),
            'obs_speed': torch.FloatTensor(speeds),
            'origin': torch.FloatTensor(origin),
            'person_id': seq['person_id'],
            'video_name': seq['video_name'],
        }


class PreprocessedETHUCYDataset(Dataset):
    """
    ETH/UCY dataset loading from preprocessed .pt files.
    
    Expected structure (from preprocess_eth_ucy.py):
        data/processed/eth_ucy/
        ├── eth/
        │   ├── train.pt
        │   ├── val.pt
        │   └── test.pt
        ├── hotel/
        │   ├── train.pt
        │   ├── val.pt
        │   └── test.pt
        └── ...
    
    Each .pt file contains:
        - obs_traj: [N, obs_len, 2]
        - pred_traj: [N, pred_len, 2]
        - obs_traj_rel: [N, obs_len, 2]
        - pred_traj_rel: [N, pred_len, 2]
        - obs_pose: [N, obs_len, 22, 3] (optional, if pose data available)
        - pred_pose: [N, pred_len, 22, 3] (optional)
    """
    
    SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        scenes: List[str] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Root directory for preprocessed ETH/UCY data
            split: 'train', 'val', or 'test'
            scenes: List of scenes to load (default: all)
            augment: Whether to apply augmentation
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.scenes = scenes or self.SCENES
        
        # Augmentation
        self.augment = augment
        self.augmentor = TrajectoryAugmentation() if augment else None
        
        # Load data
        self.obs_traj = []
        self.pred_traj = []
        self.obs_traj_rel = []
        self.pred_traj_rel = []
        self.obs_pose = []
        self.pred_pose = []
        self.scene_ids = []
        self.has_pose = False
        
        self._load_data()
        
        # Stack into tensors
        if len(self.obs_traj) > 0:
            self.obs_traj = torch.cat(self.obs_traj, dim=0)
            self.pred_traj = torch.cat(self.pred_traj, dim=0)
            self.obs_traj_rel = torch.cat(self.obs_traj_rel, dim=0)
            self.pred_traj_rel = torch.cat(self.pred_traj_rel, dim=0)
            self.scene_ids = torch.cat(self.scene_ids, dim=0)
            
            # Handle pose data
            if self.has_pose and len(self.obs_pose) > 0:
                self.obs_pose = torch.cat(self.obs_pose, dim=0)
                self.pred_pose = torch.cat(self.pred_pose, dim=0)
            else:
                self.obs_pose = None
                self.pred_pose = None
        else:
            self.obs_traj = torch.zeros(0, 8, 2)
            self.pred_traj = torch.zeros(0, 12, 2)
            self.obs_traj_rel = torch.zeros(0, 8, 2)
            self.pred_traj_rel = torch.zeros(0, 12, 2)
            self.scene_ids = torch.zeros(0, dtype=torch.long)
            self.obs_pose = None
            self.pred_pose = None
        
        pose_status = "with poses" if self.has_pose else "no poses"
        logger.info(f"PreprocessedETHUCYDataset ({split}): Loaded {len(self.obs_traj)} sequences from {len(self.scenes)} scenes ({pose_status})")
    
    def _load_data(self):
        """Load preprocessed data from .pt files."""
        for scene_idx, scene in enumerate(self.scenes):
            # Look for preprocessed file
            pt_file = os.path.join(self.data_dir, scene, f'{self.split}.pt')
            
            if not os.path.exists(pt_file):
                logger.warning(f"Preprocessed file not found: {pt_file}")
                continue
            
            # Load data
            data = torch.load(pt_file, weights_only=False)
            
            n_sequences = len(data['obs_traj'])
            if n_sequences == 0:
                continue
            
            self.obs_traj.append(data['obs_traj'])
            self.pred_traj.append(data['pred_traj'])
            self.obs_traj_rel.append(data.get('obs_traj_rel', torch.zeros_like(data['obs_traj'])))
            self.pred_traj_rel.append(data.get('pred_traj_rel', torch.zeros_like(data['pred_traj'])))
            self.scene_ids.append(torch.full((n_sequences,), scene_idx, dtype=torch.long))
            
            # Load pose data if available
            if 'obs_pose' in data and 'pred_pose' in data:
                self.obs_pose.append(data['obs_pose'])
                self.pred_pose.append(data['pred_pose'])
                self.has_pose = True
            
            logger.info(f"  Loaded {scene}/{self.split}.pt: {n_sequences} sequences")
    
    def __len__(self) -> int:
        return len(self.obs_traj)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs_traj = self.obs_traj[idx].clone()
        pred_traj = self.pred_traj[idx].clone()
        obs_traj_rel = self.obs_traj_rel[idx].clone()
        pred_traj_rel = self.pred_traj_rel[idx].clone()
        
        # Get pose if available
        obs_pose = None
        if self.has_pose and self.obs_pose is not None:
            obs_pose = self.obs_pose[idx].clone()
        
        # Apply augmentation
        if self.augment and self.augmentor is not None:
            # Convert to numpy for augmentation
            obs_np = obs_traj.numpy()
            pred_np = pred_traj.numpy()
            pose_np = obs_pose.numpy() if obs_pose is not None else None
            
            # Augmentor expects (obs_traj, pred_traj) and returns (obs, pred, pose)
            obs_np, pred_np, pose_np = self.augmentor(obs_np, pred_np, obs_pose=pose_np)
            
            # Convert back to tensor
            obs_traj = torch.FloatTensor(obs_np)
            pred_traj = torch.FloatTensor(pred_np)
            if pose_np is not None:
                obs_pose = torch.FloatTensor(pose_np)
            
            # Recompute relative
            obs_traj_rel = torch.zeros_like(obs_traj)
            obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
            pred_traj_rel = torch.zeros_like(pred_traj)
            pred_traj_rel[0] = pred_traj[0] - obs_traj[-1]
            pred_traj_rel[1:] = pred_traj[1:] - pred_traj[:-1]
        
        # Normalize around last observed position
        origin = obs_traj[-1].clone()
        obs_traj_norm = obs_traj - origin
        pred_traj_norm = pred_traj - origin
        
        # Compute velocity
        velocity = torch.zeros_like(obs_traj)
        velocity[1:] = obs_traj[1:] - obs_traj[:-1]
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        
        result = {
            'obs_traj': obs_traj_norm,
            'pred_traj': pred_traj_norm,
            'obs_traj_rel': obs_traj_rel,
            'pred_traj_rel': pred_traj_rel,
            'obs_velocity': velocity,
            'obs_speed': speed,
            'origin': origin,
            'scene_id': self.scene_ids[idx],
        }
        
        # Add pose if available
        if obs_pose is not None:
            result['obs_pose'] = obs_pose
        
        return result


class PreprocessedJTADataset(Dataset):
    """
    JTA dataset loading from preprocessed .pt files.
    
    Expected structure (from preprocess_jta_single.py):
        data/processed/jta/
        ├── train.pt
        ├── val.pt
        ├── test.pt
        └── metadata.json
    
    Each .pt file contains:
        - obs_traj: [N, obs_len, 2]
        - pred_traj: [N, pred_len, 2]
        - obs_traj_rel: [N, obs_len, 2]
        - pred_traj_rel: [N, pred_len, 2]
        - obs_pose: [N, obs_len, 22, 3]
        - pred_pose: [N, pred_len, 22, 3]
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Root directory for preprocessed JTA data
            split: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        
        # Augmentation
        self.augment = augment
        self.augmentor = TrajectoryAugmentation() if augment else None
        
        # Load data
        self._load_data()
        
        pose_status = "with poses" if self.obs_pose is not None else "no poses"
        logger.info(f"PreprocessedJTADataset ({split}): Loaded {len(self.obs_traj)} sequences ({pose_status})")
    
    def _load_data(self):
        """Load preprocessed data from .pt file."""
        pt_file = os.path.join(self.data_dir, f'{self.split}.pt')
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Preprocessed file not found: {pt_file}")
        
        data = torch.load(pt_file, weights_only=False)
        
        self.obs_traj = data['obs_traj']
        self.pred_traj = data['pred_traj']
        self.obs_traj_rel = data.get('obs_traj_rel', torch.zeros_like(self.obs_traj))
        self.pred_traj_rel = data.get('pred_traj_rel', torch.zeros_like(self.pred_traj))
        
        # Pose data
        self.obs_pose = data.get('obs_pose', None)
        self.pred_pose = data.get('pred_pose', None)
        
        # Metadata
        self.person_ids = data.get('person_ids', torch.zeros(len(self.obs_traj), dtype=torch.long))
        self.seq_ids = data.get('seq_ids', torch.zeros(len(self.obs_traj), dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.obs_traj)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs_traj = self.obs_traj[idx].clone()
        pred_traj = self.pred_traj[idx].clone()
        obs_traj_rel = self.obs_traj_rel[idx].clone()
        pred_traj_rel = self.pred_traj_rel[idx].clone()
        
        # Get pose if available
        obs_pose = None
        if self.obs_pose is not None:
            obs_pose = self.obs_pose[idx].clone()
        
        # Apply augmentation
        if self.augment and self.augmentor is not None:
            obs_np = obs_traj.numpy()
            pred_np = pred_traj.numpy()
            pose_np = obs_pose.numpy() if obs_pose is not None else None
            
            obs_np, pred_np, pose_np = self.augmentor(obs_np, pred_np, obs_pose=pose_np)
            
            obs_traj = torch.FloatTensor(obs_np)
            pred_traj = torch.FloatTensor(pred_np)
            if pose_np is not None:
                obs_pose = torch.FloatTensor(pose_np)
            
            # Recompute relative trajectories
            obs_traj_rel = torch.zeros_like(obs_traj)
            obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
            pred_traj_rel = torch.zeros_like(pred_traj)
            pred_traj_rel[0] = pred_traj[0] - obs_traj[-1]
            pred_traj_rel[1:] = pred_traj[1:] - pred_traj[:-1]
        
        # Normalize around last observed position
        origin = obs_traj[-1].clone()
        obs_traj_norm = obs_traj - origin
        pred_traj_norm = pred_traj - origin
        
        # Compute velocity
        velocity = torch.zeros_like(obs_traj)
        velocity[1:] = obs_traj[1:] - obs_traj[:-1]
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        
        result = {
            'obs_traj': obs_traj_norm,
            'pred_traj': pred_traj_norm,
            'obs_traj_rel': obs_traj_rel,
            'pred_traj_rel': pred_traj_rel,
            'obs_velocity': velocity,
            'obs_speed': speed,
            'origin': origin,
            'person_id': self.person_ids[idx],
            'seq_id': self.seq_ids[idx],
        }
        
        # Add pose if available
        if obs_pose is not None:
            result['obs_pose'] = obs_pose
        
        return result


class ETHUCYDataset(Dataset):
    """
    ETH/UCY dataset from TrajPRed format.
    
    Expected structure:
        eth_ucy/
        ├── eth/{train,val,test}/biwi_eth.txt
        ├── hotel/{train,val,test}/...
        ├── univ/{train,val,test}/...
        ├── zara1/{train,val,test}/...
        └── zara2/{train,val,test}/...
    
    Data format (space/tab separated):
        frame_id    pedestrian_id    x    y
    """
    
    SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        scenes: List[str] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        delim: str = None,  # Auto-detect
        augment: bool = False,
        filter_static: bool = True,
        static_threshold: float = 0.1,
    ):
        """
        Args:
            data_dir: Root directory for ETH/UCY data
            split: 'train', 'val', or 'test'
            scenes: List of scenes to load (default: all)
            obs_len: Observation length
            pred_len: Prediction length
            skip: Sliding window stride
            min_ped: Minimum pedestrians per scene
            delim: Delimiter (auto-detect if None)
            augment: Whether to apply augmentation
            filter_static: Filter out static pedestrians
            static_threshold: Min displacement to be non-static
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.scenes = scenes or self.SCENES
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.delim = delim
        self.filter_static = filter_static
        self.static_threshold = static_threshold
        
        # Augmentation
        self.augment = augment
        self.augmentor = TrajectoryAugmentation() if augment else None
        
        # Load sequences
        self.sequences = []
        self._load_data()
        
        logger.info(f"ETHUCYDataset ({split}): Loaded {len(self.sequences)} sequences from {len(self.scenes)} scenes")
    
    def _load_data(self):
        """Load data from all specified scenes."""
        for scene in self.scenes:
            scene_dir = os.path.join(self.data_dir, scene, self.split)
            
            if not os.path.exists(scene_dir):
                logger.warning(f"Scene directory not found: {scene_dir}")
                continue
            
            txt_files = [f for f in os.listdir(scene_dir) if f.endswith('.txt')]
            
            for txt_file in txt_files:
                filepath = os.path.join(scene_dir, txt_file)
                sequences = self._process_trajectory_file(filepath, scene)
                self.sequences.extend(sequences)
    
    def _process_trajectory_file(self, filepath: str, scene: str) -> List[Dict]:
        """Process a single trajectory file."""
        # Read data
        data = self._read_file(filepath)
        
        if data is None or len(data) == 0:
            return []
        
        # Group by pedestrian
        frames = np.unique(data[:, 0]).tolist()
        frame_data = {}
        for frame in frames:
            frame_data[frame] = data[data[:, 0] == frame][:, 1:]  # [ped_id, x, y]
        
        sequences = []
        
        # Create sequences with sliding window
        for start_idx in range(0, len(frames) - self.seq_len + 1, self.skip):
            seq_frames = frames[start_idx:start_idx + self.seq_len]
            
            # Find pedestrians present in all frames
            peds_in_frames = [set(frame_data[f][:, 0]) for f in seq_frames]
            common_peds = set.intersection(*peds_in_frames)
            
            if len(common_peds) < self.min_ped:
                continue
            
            # Extract trajectory for each pedestrian
            for ped_id in common_peds:
                trajectory = []
                
                for frame in seq_frames:
                    ped_data = frame_data[frame]
                    idx = np.where(ped_data[:, 0] == ped_id)[0]
                    if len(idx) > 0:
                        trajectory.append(ped_data[idx[0], 1:3])  # x, y
                
                if len(trajectory) != self.seq_len:
                    continue
                
                trajectory = np.array(trajectory)
                
                # Filter static pedestrians
                if self.filter_static:
                    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
                    if displacement < self.static_threshold:
                        continue
                
                sequences.append({
                    'trajectory': trajectory,
                    'person_id': int(ped_id),
                    'scene': scene,
                })
        
        return sequences
    
    def _read_file(self, filepath: str) -> Optional[np.ndarray]:
        """Read trajectory file with auto-delimiter detection."""
        try:
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    
                    # Auto-detect delimiter
                    if self.delim is None:
                        if '\t' in line:
                            delim = '\t'
                        else:
                            delim = None  # Split on whitespace
                    else:
                        delim = self.delim
                    
                    parts = line.split(delim)
                    if len(parts) >= 4:
                        # frame_id, ped_id, x, y
                        data.append([float(p) for p in parts[:4]])
            
            return np.array(data) if len(data) > 0 else None
            
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        trajectory = seq['trajectory'].copy()
        
        # Split
        obs_traj = trajectory[:self.obs_len]
        pred_traj = trajectory[self.obs_len:]
        
        # Apply augmentation
        if self.augment and self.augmentor is not None:
            obs_traj, pred_traj, _ = self.augmentor(obs_traj, pred_traj)
        
        # Normalize
        origin = obs_traj[-1].copy()
        obs_traj_norm = obs_traj - origin
        pred_traj_norm = pred_traj - origin
        
        # Velocity
        velocities = compute_velocity(np.vstack([obs_traj_norm, pred_traj_norm[:1]]))
        speeds = compute_speed(velocities)
        
        return {
            'obs_traj': torch.FloatTensor(obs_traj_norm),
            'pred_traj': torch.FloatTensor(pred_traj_norm),
            'obs_velocity': torch.FloatTensor(velocities),
            'obs_speed': torch.FloatTensor(speeds),
            'origin': torch.FloatTensor(origin),
            'person_id': seq['person_id'],
            'scene': seq['scene'],
        }


class CombinedDataset(Dataset):
    """
    Combines multiple trajectory datasets (e.g., multiple ETH/UCY scenes).
    """
    
    def __init__(self, datasets: List[Dataset]):
        """
        Args:
            datasets: List of datasets to combine
        """
        super().__init__()
        self.datasets = datasets
        
        # Compute cumulative lengths
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                break
        
        # Compute local index
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching trajectory data.
    
    Handles variable-length neighbor data if present.
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            collated[key] = values  # Keep as list
        else:
            collated[key] = values  # Keep as list
    
    return collated


# =============================================================================
# Factory Functions
# =============================================================================

def create_jta_datasets(
    data_dir: str,
    obs_len: int = 8,
    pred_len: int = 12,
    **kwargs
) -> Tuple[JTADataset, JTADataset, JTADataset]:
    """
    Create train, val, test datasets for JTA.
    
    Args:
        data_dir: JTA root directory
        obs_len: Observation length
        pred_len: Prediction length
        **kwargs: Additional arguments for JTADataset
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_ds = JTADataset(
        data_dir, split='train', obs_len=obs_len, pred_len=pred_len,
        augment=True, **kwargs
    )
    val_ds = JTADataset(
        data_dir, split='val', obs_len=obs_len, pred_len=pred_len,
        augment=False, **kwargs
    )
    test_ds = JTADataset(
        data_dir, split='test', obs_len=obs_len, pred_len=pred_len,
        augment=False, **kwargs
    )
    
    return train_ds, val_ds, test_ds


def create_eth_ucy_datasets(
    data_dir: str,
    obs_len: int = 8,
    pred_len: int = 12,
    scenes: List[str] = None,
    **kwargs
) -> Tuple[ETHUCYDataset, ETHUCYDataset, ETHUCYDataset]:
    """
    Create train, val, test datasets for ETH/UCY.
    
    Args:
        data_dir: ETH/UCY root directory
        obs_len: Observation length
        pred_len: Prediction length
        scenes: List of scenes to load
        **kwargs: Additional arguments for ETHUCYDataset
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_ds = ETHUCYDataset(
        data_dir, split='train', scenes=scenes,
        obs_len=obs_len, pred_len=pred_len, augment=True, **kwargs
    )
    val_ds = ETHUCYDataset(
        data_dir, split='val', scenes=scenes,
        obs_len=obs_len, pred_len=pred_len, augment=False, **kwargs
    )
    test_ds = ETHUCYDataset(
        data_dir, split='test', scenes=scenes,
        obs_len=obs_len, pred_len=pred_len, augment=False, **kwargs
    )
    
    return train_ds, val_ds, test_ds


def create_eth_ucy_leave_one_out(
    data_dir: str,
    test_scene: str,
    obs_len: int = 8,
    pred_len: int = 12,
    val_ratio: float = 0.15,
    **kwargs
) -> Tuple[PreprocessedETHUCYDataset, PreprocessedETHUCYDataset, PreprocessedETHUCYDataset]:
    """
    Create leave-one-out cross-validation datasets for ETH/UCY.
    
    Uses preprocessed .pt files from the test_scene folder.
    The preprocessor already sets up leave-one-out structure:
    - eth/train.pt contains data from hotel, univ, zara1, zara2
    - eth/test.pt contains only eth data
    
    Args:
        data_dir: ETH/UCY root directory (data/processed/eth_ucy)
        test_scene: Scene to use for testing
        obs_len: Observation length (not used, already set in preprocessing)
        pred_len: Prediction length (not used, already set in preprocessing)
        val_ratio: Fraction of training data to use for validation (not used)
        **kwargs: Additional arguments
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # The preprocessor already created leave-one-out structure
    # Under each scene folder (eth, hotel, etc.), train.pt has data from OTHER scenes
    # and test.pt has data from THAT scene only
    
    # Load from the test_scene folder (which has the correct LOO split)
    train_ds = PreprocessedETHUCYDataset(
        data_dir, 
        split='train', 
        scenes=[test_scene],  # Load train.pt from test_scene folder
        augment=True
    )
    
    val_ds = PreprocessedETHUCYDataset(
        data_dir, 
        split='val', 
        scenes=[test_scene],  # Load val.pt from test_scene folder
        augment=False
    )
    
    test_ds = PreprocessedETHUCYDataset(
        data_dir, 
        split='test', 
        scenes=[test_scene],  # Load test.pt from test_scene folder
        augment=False
    )
    
    return train_ds, val_ds, test_ds