"""Custom DataLoader factory with collate for variable-length sequences."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import VideoSummarizationDataset
from .feature_loader import FeatureLoader
from .label_loader import LabelLoader


def collate_temporal_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Collate batch: (B, T, D), (B, T), (B,) lengths, list of video_ids.
    """
    feats = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    lengths = torch.tensor([x[2] for x in batch], dtype=torch.long)
    video_ids = [x[3] for x in batch]
    return feats, labels, lengths, video_ids


def create_dataloaders(
    video_ids: List[str],
    feature_loader: FeatureLoader,
    label_loader: LabelLoader,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 8,
    num_workers: int = 0,
    max_seq_len: int = 320,
    min_seq_len: int = 16,
    padding_value: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with reproducible split.
    """
    full = VideoSummarizationDataset(
        video_ids=video_ids,
        feature_loader=feature_loader,
        label_loader=label_loader,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        padding_value=padding_value,
    )
    n = len(full)
    if n == 0:
        raise ValueError("No valid samples after filtering")
    gen = torch.Generator().manual_seed(seed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_temporal_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_temporal_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_temporal_batch,
    )
    return train_loader, val_loader, test_loader
