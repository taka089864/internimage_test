#!/usr/bin/env python
"""Test single-head training to reproduce the error."""

import sys
sys.path.append('.')

import mmcv_custom  # noqa: F401
import mmseg_custom  # noqa: F401
import tb_multi_lr_hook  # noqa: F401

from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

# Load config
cfg = Config.fromfile('configs/medical_shift/internimage_l_640_teeth_single.py')

# Set minimal training config
cfg.work_dir = './test_work_dir'
cfg.gpu_ids = [0]
cfg.device = 'cuda'
cfg.runner.max_iters = 10  # Just run 10 iterations for testing
cfg.checkpoint_config.interval = 100  # Don't save checkpoints
cfg.log_config.interval = 1
cfg.evaluation = None  # Skip evaluation

# Build dataset
print("Building dataset...")
datasets = [build_dataset(cfg.data.train)]

# Build model
print("Building model...")
model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# Try to train
print("Starting training...")
try:
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=None,
        meta=dict()
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()