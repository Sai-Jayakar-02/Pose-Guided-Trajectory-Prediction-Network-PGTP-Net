"""
Data loading and preprocessing module for trajectory prediction.

Supports:
- JTA dataset (native 3D pose annotations, 22 joints)
- ETH/UCY dataset (TrajPRed format with train/val/test splits)
- Custom datasets

Usage:
    from src.data import JTADataset, ETHUCYDataset, create_dataloaders
    
    # JTA
    train_ds = JTADataset('data/raw/jta', split='train')
    
    # ETH/UCY
    train_ds = ETHUCYDataset('data/raw/eth_ucy', split='train')
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_jta_dataloaders('data/raw/jta')
"""

# Dataset classes
from .datasets import (
    TrajectoryDataset,
    JTADataset,
    ETHUCYDataset,
    PreprocessedETHUCYDataset,
    PreprocessedJTADataset,
    CombinedDataset,
    collate_fn,
    # Factory functions
    create_jta_datasets,
    create_eth_ucy_datasets,
    create_eth_ucy_leave_one_out,
    # Constants
    JTA_JOINTS,
    JTA_NUM_JOINTS,
    JTA_PELVIS_IDX,
    JTA_KEY_JOINTS,
    JTA_FLIP_PAIRS,
    JTA_CAMERA,
)

# Preprocessing utilities
from .preprocessing import (
    normalize_trajectory,
    denormalize_trajectory,
    normalize_pose,
    compute_velocity,
    compute_speed,
    compute_acceleration,
    compute_heading,
    compute_heading_change,
    create_sequences,
    world_to_pixel,
    pixel_to_world,
    interpolate_trajectory,
    resample_by_distance,
    compute_social_features,
    filter_static_pedestrians,
    smooth_trajectory,
    # Constants
    JTA_NUM_JOINTS,
    JTA_PELVIS_IDX,
    JTA_KEY_JOINTS,
    JTA_FLIP_PAIRS,
    COCO_NUM_JOINTS,
    COCO_KEY_JOINTS,
    COCO_FLIP_PAIRS,
)

# Augmentation
from .augmentation import (
    TrajectoryAugmentation,
    PoseAugmentation,
    mixup_trajectories,
    cutout_trajectory,
)

# DataLoader utilities
from .dataloaders import (
    create_dataloader,
    create_dataloaders,
    create_jta_dataloaders,
    create_preprocessed_jta_dataloaders,
    create_eth_ucy_dataloaders,
    create_leave_one_out_dataloaders,
    get_dataloader_from_config,
    InfiniteDataLoader,
    MultiEpochsDataLoader,
    worker_init_fn,
    get_sample_batch,
    print_batch_info,
)

__all__ = [
    # Datasets
    'TrajectoryDataset',
    'JTADataset',
    'ETHUCYDataset',
    'PreprocessedETHUCYDataset',
    'PreprocessedJTADataset',
    'CombinedDataset',
    'collate_fn',
    'create_jta_datasets',
    'create_eth_ucy_datasets',
    'create_eth_ucy_leave_one_out',
    
    # Preprocessing
    'normalize_trajectory',
    'denormalize_trajectory',
    'normalize_pose',
    'compute_velocity',
    'compute_speed',
    'compute_acceleration',
    'compute_heading',
    'compute_heading_change',
    'create_sequences',
    'world_to_pixel',
    'pixel_to_world',
    'interpolate_trajectory',
    'resample_by_distance',
    'compute_social_features',
    'filter_static_pedestrians',
    'smooth_trajectory',
    
    # Augmentation
    'TrajectoryAugmentation',
    'PoseAugmentation',
    'mixup_trajectories',
    'cutout_trajectory',
    
    # DataLoaders
    'create_dataloader',
    'create_dataloaders',
    'create_jta_dataloaders',
    'create_eth_ucy_dataloaders',
    'create_leave_one_out_dataloaders',
    'get_dataloader_from_config',
    'InfiniteDataLoader',
    'MultiEpochsDataLoader',
    'worker_init_fn',
    'get_sample_batch',
    'print_batch_info',
    
    # Constants
    'JTA_JOINTS',
    'JTA_NUM_JOINTS',
    'JTA_PELVIS_IDX',
    'JTA_KEY_JOINTS',
    'JTA_FLIP_PAIRS',
    'JTA_CAMERA',
    'COCO_NUM_JOINTS',
    'COCO_KEY_JOINTS',
    'COCO_FLIP_PAIRS',
]