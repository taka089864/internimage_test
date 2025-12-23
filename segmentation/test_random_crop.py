#!/usr/bin/env python
"""Test RandomCrop behavior to trigger different sizes."""

import sys
sys.path.append('.')

import mmcv_custom  # noqa: F401
import mmseg_custom  # noqa: F401

import numpy as np
from mmcv import Config
from mmseg.datasets import build_dataset
from torch.utils.data import DataLoader
from mmcv.parallel import collate

# Load single-head config
cfg = Config.fromfile('configs/medical_shift/internimage_l_640_teeth_single.py')

# Build dataset
dataset = build_dataset(cfg.data.train)
print(f'Dataset built: {len(dataset)} samples')

# Test multiple samples to find different crop sizes
print("\n=== Looking for variable crop sizes ===")
sizes_found = set()
for i in range(100):  # Check first 100 samples
    sample = dataset[i]
    size = sample["img"].data.shape[-2:]
    sizes_found.add(size)
    if len(sizes_found) > 1:
        print(f"Found different sizes at sample {i}: {sizes_found}")
        break

if len(sizes_found) == 1:
    print(f"All samples have the same size: {sizes_found}")
    
# Force different sizes by modifying cat_max_ratio
print("\n=== Testing with cat_max_ratio=0.5 ===")
cfg.data.train.pipeline[3]['cat_max_ratio'] = 0.5  # RandomCrop
dataset2 = build_dataset(cfg.data.train)

# Check sizes again
sizes_found2 = set()
for i in range(100):
    sample = dataset2[i]
    size = sample["img"].data.shape[-2:]
    sizes_found2.add(size)
    
print(f"Sizes found with cat_max_ratio=0.5: {sizes_found2}")

# Test DataLoader with potentially different sizes
if len(sizes_found2) > 1:
    print("\n=== Testing DataLoader with different sizes ===")
    try:
        dataloader = DataLoader(
            dataset2,
            batch_size=2,
            num_workers=0,
            collate_fn=lambda batch: collate(batch, samples_per_gpu=2),
            shuffle=False  # Ensure we get consistent samples
        )
        
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: img shape {batch['img'].data[0].shape}")
            if i >= 2:
                break
                
    except Exception as e:
        print(f"Error in DataLoader: {type(e).__name__}: {e}")