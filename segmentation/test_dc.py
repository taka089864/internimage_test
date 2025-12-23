#!/usr/bin/env python
"""Test DataContainer behavior."""

import torch
import numpy as np
from mmcv.parallel import DataContainer as DC

# Test stacking different sizes
print("=== Testing DataContainer stacking ===")

# Create tensors of different sizes
tensor1 = torch.randn(3, 623, 623)
tensor2 = torch.randn(3, 640, 640)

# Create DataContainers with stack=True
dc1 = DC(tensor1, stack=True)
dc2 = DC(tensor2, stack=True)

print(f"DC1 stack: {dc1.stack}, pad_dims: {dc1.pad_dims}")
print(f"DC2 stack: {dc2.stack}, pad_dims: {dc2.pad_dims}")

# Test what happens when we try to collate them
from torch.utils.data._utils.collate import default_collate

try:
    batch = default_collate([{'img': dc1}, {'img': dc2}])
    print("Success! Batch shape:", batch['img'].data.shape)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also test with pad_dims
print("\n=== Testing with pad_dims ===")
dc1_pad = DC(tensor1, stack=True, pad_dims=2)
dc2_pad = DC(tensor2, stack=True, pad_dims=2)

try:
    batch = default_collate([{'img': dc1_pad}, {'img': dc2_pad}])
    print("Success! Batch shape:", batch['img'].data.shape)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")