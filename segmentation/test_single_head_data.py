#!/usr/bin/env python
"""Test single-head data loading with proper imports."""

import sys
sys.path.append('.')

import mmcv_custom  # noqa: F401
import mmseg_custom  # noqa: F401

from mmcv import Config
from mmseg.datasets import build_dataset
from torch.utils.data import DataLoader
from mmcv.parallel import collate

# Load single-head config
cfg = Config.fromfile('configs/medical_shift/internimage_l_640_teeth_single.py')

# Build dataset
try:
    dataset = build_dataset(cfg.data.train)
    print(f'Dataset built successfully: {len(dataset)} samples')
    
    # Check a few samples
    print("\n=== Checking individual samples ===")
    for i in [0, 1, 2]:
        sample = dataset[i]
        print(f'\nSample {i}:')
        print(f'  img shape: {sample["img"].data.shape}')
        print(f'  img pad_dims: {sample["img"].pad_dims}')
        print(f'  gt_semantic_seg shape: {sample["gt_semantic_seg"].data.shape}')
        print(f'  gt_semantic_seg pad_dims: {sample["gt_semantic_seg"].pad_dims}')
    
    # Test DataLoader with collate
    print("\n=== Testing DataLoader ===")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, samples_per_gpu=2)
    )
    
    # Get first batch
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  img type: {type(batch['img'])}")
        print(f"  img.data type: {type(batch['img'].data)}")
        if isinstance(batch['img'].data, list):
            print(f"  img.data[0] shape: {batch['img'].data[0].shape}")
        else:
            print(f"  img.data shape: {batch['img'].data.shape}")
        print(f"  gt_semantic_seg.data[0] shape: {batch['gt_semantic_seg'].data[0].shape}")
        
        if i >= 2:  # Check first 3 batches
            break
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()