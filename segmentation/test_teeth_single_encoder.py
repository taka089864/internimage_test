#!/usr/bin/env python
"""Test single-head with EncoderDecoderForTeethSingle."""

import sys
sys.path.append('.')

import mmcv_custom  # noqa: F401
import mmseg_custom  # noqa: F401

from mmcv import Config
from mmseg.models import build_segmentor

# Load config
cfg = Config.fromfile('configs/medical_shift/internimage_l_640_teeth_single.py')

# Build model
print("Building model...")
try:
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    print(f"Model built successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Model class: {model.__class__}")
    
    # Check if model has expected attributes
    print(f"\nModel attributes:")
    print(f"  has backbone: {hasattr(model, 'backbone')}")
    print(f"  has decode_head: {hasattr(model, 'decode_head')}")
    if hasattr(model, 'decode_head'):
        print(f"  decode_head type: {type(model.decode_head).__name__}")
        print(f"  decode_head num_classes: {model.decode_head.num_classes}")
    
except Exception as e:
    print(f"Error building model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()