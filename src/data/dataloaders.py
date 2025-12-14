"""
DataLoader factory for trajectory prediction datasets.

Provides convenient functions to create DataLoaders with proper
configuration for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Tuple, Optional, Dict, List, Union
import logging

from .datasets import (
    JTADataset,
    ETHUCYDataset,
    CombinedDataset,
    collate_fn,
    create_jta_datasets,
    create_eth_ucy_datasets,
    create_eth_ucy_leave_one_out,
)

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a DataLoader with common settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, val, and test DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
    
    Returns:
        (train_loader, val_loader, test_loader or None)
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    logger.info(f"Created DataLoaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches"
                f"{f', Test: {len(test_loader)} batches' if test_loader else ''}")
    
    return train_loader, val_loader, test_loader


def create_jta_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    obs_len: int = 8,
    pred_len: int = 12,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for JTA dataset.
    
    Args:
        data_dir: JTA root directory
        batch_size: Batch size
        obs_len: Observation length
        pred_len: Prediction length
        num_workers: Number of workers
        **kwargs: Additional args for JTADataset
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = create_jta_datasets(
        data_dir, obs_len=obs_len, pred_len=pred_len, **kwargs
    )
    
    return create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers
    )


def create_preprocessed_jta_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for preprocessed JTA dataset.
    
    Args:
        data_dir: Directory containing train.pt, val.pt, test.pt
        batch_size: Batch size
        num_workers: Number of workers
        augment_train: Whether to augment training data
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from .datasets import PreprocessedJTADataset
    
    train_ds = PreprocessedJTADataset(data_dir, split='train', augment=augment_train)
    val_ds = PreprocessedJTADataset(data_dir, split='val', augment=False)
    
    # Test split might not exist
    try:
        test_ds = PreprocessedJTADataset(data_dir, split='test', augment=False)
    except FileNotFoundError:
        test_ds = val_ds  # Use val as test if test doesn't exist
    
    return create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers
    )


def create_eth_ucy_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    obs_len: int = 8,
    pred_len: int = 12,
    scenes: List[str] = None,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for ETH/UCY dataset.
    
    Args:
        data_dir: ETH/UCY root directory
        batch_size: Batch size
        obs_len: Observation length
        pred_len: Prediction length
        scenes: List of scenes to load
        num_workers: Number of workers
        **kwargs: Additional args for ETHUCYDataset
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = create_eth_ucy_datasets(
        data_dir, obs_len=obs_len, pred_len=pred_len, scenes=scenes, **kwargs
    )
    
    return create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers
    )


def create_leave_one_out_dataloaders(
    data_dir: str,
    test_scene: str,
    batch_size: int = 64,
    obs_len: int = 8,
    pred_len: int = 12,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for ETH/UCY leave-one-out cross-validation.
    
    Args:
        data_dir: ETH/UCY root directory
        test_scene: Scene to hold out for testing
        batch_size: Batch size
        obs_len: Observation length
        pred_len: Prediction length
        num_workers: Number of workers
        **kwargs: Additional args for ETHUCYDataset
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = create_eth_ucy_leave_one_out(
        data_dir, test_scene=test_scene,
        obs_len=obs_len, pred_len=pred_len, **kwargs
    )
    
    return create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers
    )


def get_dataloader_from_config(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from configuration dictionary.
    
    Args:
        config: Configuration dictionary with keys:
            - dataset.name: 'jta' or 'eth_ucy'
            - dataset.data_dir: Data directory
            - training.batch_size: Batch size
            - sequence.obs_len: Observation length
            - sequence.pred_len: Prediction length
            - training.num_workers: Number of workers
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_name = config.get('dataset', {}).get('name', 'eth_ucy')
    data_dir = config.get('paths', {}).get('raw_data', 'data/raw')
    
    batch_size = config.get('training', {}).get('batch_size', 64)
    num_workers = config.get('training', {}).get('num_workers', 4)
    obs_len = config.get('sequence', {}).get('obs_len', 8)
    pred_len = config.get('sequence', {}).get('pred_len', 12)
    
    if dataset_name == 'jta':
        data_path = f"{data_dir}/jta"
        return create_jta_dataloaders(
            data_path, batch_size=batch_size,
            obs_len=obs_len, pred_len=pred_len,
            num_workers=num_workers
        )
    
    elif dataset_name == 'eth_ucy':
        data_path = f"{data_dir}/eth_ucy"
        
        # Check for leave-one-out config
        test_scene = config.get('dataset', {}).get('eth_ucy', {}).get('test_scene')
        
        if test_scene:
            return create_leave_one_out_dataloaders(
                data_path, test_scene=test_scene,
                batch_size=batch_size,
                obs_len=obs_len, pred_len=pred_len,
                num_workers=num_workers
            )
        else:
            return create_eth_ucy_dataloaders(
                data_path, batch_size=batch_size,
                obs_len=obs_len, pred_len=pred_len,
                num_workers=num_workers
            )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class InfiniteDataLoader:
    """
    DataLoader that cycles infinitely.
    Useful for training with iteration-based (not epoch-based) loops.
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        Args:
            dataloader: Base DataLoader
        """
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def __len__(self):
        return len(self.dataloader)


class MultiEpochsDataLoader(DataLoader):
    """
    DataLoader that maintains worker processes across epochs.
    Reduces initialization overhead for multi-epoch training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    
    def __len__(self):
        return len(self.batch_sampler.sampler)
    
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever."""
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def worker_init_fn(worker_id: int):
    """
    Initialize worker with unique random seed.
    
    This ensures different workers produce different random augmentations.
    """
    import numpy as np
    import random
    
    # Each worker gets a unique seed based on base seed + worker_id
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_sample_batch(dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch from a DataLoader.
    Useful for debugging and model initialization.
    
    Args:
        dataloader: DataLoader to sample from
    
    Returns:
        Single batch of data
    """
    for batch in dataloader:
        return batch
    raise RuntimeError("DataLoader is empty")


def print_batch_info(batch: Dict[str, torch.Tensor]):
    """
    Print information about a batch for debugging.
    
    Args:
        batch: Batch dictionary from DataLoader
    """
    print("=" * 50)
    print("Batch Information:")
    print("=" * 50)
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")
    
    print("=" * 50)