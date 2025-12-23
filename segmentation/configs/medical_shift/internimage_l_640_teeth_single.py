# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Single Head版：う蝕のみを直接セグメンテーション (mmengine format)
_base_ = [
    '../_base_/datasets/teeth_single.py',  # 新しいデータセット設定
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

import inspect
import os.path as osp


def _load_base_teeth_cfg():
    cfg_file = inspect.getfile(lambda: None)
    cfg_dir = osp.dirname(cfg_file)
    base_path = osp.join(cfg_dir, '../_base_/datasets/teeth_single.py')
    namespace = {}
    with open(base_path, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), base_path, 'exec'), namespace)
    namespace.pop('__builtins__', None)
    keys = ('data_root', 'class_color_json', 'caries_color_json')
    return {k: namespace.get(k) for k in keys}


_base_teeth_cfg = _load_base_teeth_cfg()
data_root = _base_teeth_cfg.get('data_root')
class_color_json = _base_teeth_cfg.get('class_color_json')
caries_color_json = _base_teeth_cfg.get('caries_color_json')
train_list = osp.join(data_root, 'train.list') if data_root else None
val_list = osp.join(data_root, 'valid.list') if data_root else None

# mmengine config
default_scope = 'mmseg'

# Custom imports to ensure custom hooks are registered
# This allows the config to be used from any entry point
custom_imports = dict(
    imports=['mmseg_custom.hooks.iter_logger_hook', 'mmseg_custom.hooks.clear_cuda_cache_hook'],
    allow_failed_imports=False)

pretrained = 'checkpoint/upernet_internimage_l_640_160k_ade20k.pth'

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoderForTeethSingle',  # single head専用のEncoderDecoderForTeethSingleを使用
    
    backbone=dict(
        type='InternImage',
        core_op='DCNv3_pytorch',
        channels=160,  # Large: 160
        depths=[5, 5, 22, 5],  # Large: [5, 5, 22, 5]
        groups=[10, 20, 40, 80],  # Large: [10, 20, 40, 80]
        mlp_ratio=4.,
        drop_path_rate=0.4,  # Large: 0.4
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,  # Large: 2.0
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.')
    ),

    decode_head=dict(
        type='TeethSingleHead',  # 新しいsingle head
        in_channels=[160, 320, 640, 1280],  # Large
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=128,
        dropout_ratio=0.1,

        # num_classes=5, # 0: background, 1: A1 confirmed, 2: A1 suspect, 3: A2 confirmed, 4: A2 suspect
        num_classes=3,  # 0: background, 1: confirmed, 2: suspect

        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='TeethSingleHeadLoss', loss_weight=1.0)),
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Delete old config fields
data = dict(_delete_=True)
optimizer = dict(_delete_=True)
lr_config = dict(_delete_=True)
runner = dict(_delete_=True)
checkpoint_config = dict(_delete_=True)
evaluation = dict(_delete_=True)

# ============================================================
# Data Loaders (mmengine format)
# ============================================================
train_dataloader = dict(
    batch_size=12,
    # batch_size=4,

    num_workers=8,
    # num_workers=0,

    # add
    prefetch_factor=4, # 追加
    pin_memory=True,

    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='TeethSingleDataset',
        data_root=data_root,  # 親コンフィグの変数を参照
        data_list=train_list,  # 親の変数を使って構築
        class_color_json=class_color_json,  # 親コンフィグの変数を参照
        caries_color_json=caries_color_json,  # 親コンフィグの変数を参照
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTeethAnnotations'),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', mean=[148.0188864091492, 148.0188864091492, 148.0188864091492],
                 std=[57.24381990728073, 57.24381990728073, 57.24381990728073], to_rgb=False),
            dict(type='PackSegInputs'),
        ]
    )
)

val_dataloader = dict(
    batch_size=1,

    num_workers=4,
    # num_workers=1,

    # add
    prefetch_factor=4,
    pin_memory=True,

    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='TeethSingleDataset',
        data_root=data_root,  # 親コンフィグの変数を参照
        data_list=val_list,  # 親の変数を使って構築
        class_color_json=class_color_json,  # 親コンフィグの変数を参照
        caries_color_json=caries_color_json,  # 親コンフィグの変数を参照
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTeethAnnotations'),  # validation時もGTをロード
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Normalize', mean=[148.0188864091492, 148.0188864091492, 148.0188864091492],
                 std=[57.24381990728073, 57.24381990728073, 57.24381990728073], to_rgb=False),
            dict(type='PackSegInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction'))
        ]
    )
)

test_dataloader = val_dataloader

# Override test_pipeline for mmseg 1.2.2 compatibility (MultiScaleFlipAug, Collect removed)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Normalize', mean=[148.0188864091492, 148.0188864091492, 148.0188864091492],
         std=[57.24381990728073, 57.24381990728073, 57.24381990728073], to_rgb=False),
    dict(type='PackSegInputs'),
]

# ============================================================
# Optimizer & Scheduler (mmengine format)
# ============================================================
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.00002,  # Large: 0.00002
#         betas=(0.9, 0.999),
#         weight_decay=0.05
#     ),
#     constructor='CustomLayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(
#         num_layers=37,  # Large: 37 (5+5+22+5)
#         layer_decay_rate=0.94,  # Large: 0.94
#         depths=[5, 5, 22, 5],
#         offset_lr_scale=1.0
#     )
# )

# ============================================================
# Optimizer & Scheduler (mmengine format)
# ============================================================
# AMP (Automatic Mixed Precision) 設定
# - FP16とFP32を混在させることで学習を高速化（20-30%の高速化）
# - メモリ使用量を約50%削減
# - 動的損失スケーリングでFP16のアンダーフロー/オーバーフローを防止
optim_wrapper = dict(
    type='AmpOptimWrapper',  # OptimWrapperからAmpOptimWrapperに変更
    optimizer=dict(
        type='AdamW',
        lr=0.00002,  # Large: 0.00002
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=37,  # Large: 37 (5+5+22+5)
        layer_decay_rate=0.94,  # Large: 0.94
        depths=[5, 5, 22, 5],
        offset_lr_scale=1.0
    ),

    # 勾配クリップ（公式設定に準拠、大モデル+AMP時の発散防止）
    # clip_grad=dict(max_norm=0.1, norm_type=2),

    # AMP設定
    loss_scale='dynamic',  # 動的損失スケーリング（推奨）
    # loss_scale=512.0,    # 固定スケーリングを使う場合はこちら
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False
    )
]

# ============================================================
# Training Configuration (mmengine format)
# ============================================================
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=160000,
    # val_interval=1000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================
# Hooks Configuration (mmengine format)
# ============================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='IterLoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,

        # interval=100,
        interval=1000,

        max_keep_ckpts=80,
        save_best='mDice',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Custom hooks
custom_hooks = [
    dict(type='TensorboardMultiLrHook', priority='VERY_LOW'),
    dict(type='ClearCUDACacheHook', priority='VERY_LOW')
]

# ============================================================
# Evaluation Configuration (mmengine format)
# ============================================================
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    dataset_meta=dict(
        classes=('bg', 'A1_tai_caries_confirmed', 'A1_tai_caries_suspect'),
        palette=[[0, 0, 0], [255, 0, 143], [123, 236, 0]]
    )
)
test_evaluator = val_evaluator

# ============================================================
# Runtime Configuration
# ============================================================
# Random seed
randomness = dict(seed=42, deterministic=False)

# Environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Log processor
log_processor = dict(by_epoch=False)
log_level = 'INFO'
