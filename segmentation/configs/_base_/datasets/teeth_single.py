# dataset settings for single head (caries only)
dataset_type = 'TeethSingleDataset'

data_root = 'path_to_data_root'
class_color_json ='path_to/teeth_colors.json'
caries_color_json ='path_to/caries_colors.json'

# medical_shift_teeth
img_norm_cfg = dict(
    mean=[148.0188864091492, 148.0188864091492, 148.0188864091492],
    std=[57.24381990728073, 57.24381990728073, 57.24381990728073],
    to_rgb=False)

# crop_size = (800, 800)
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTeethAnnotations'),  # う蝕のみを読み込む

    dict(type='Resize', scale=crop_size, ratio_range=(0.9, 1.1), keep_ratio=True),  # mmseg 1.2.2: img_scale -> scale
    # dict(type='Resize', scale=(640, 640), ratio_range=(0.9, 1.1), keep_ratio=True),

    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # 医療画像では色調変化を無効化
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='TeethFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',

        scale=crop_size,  # mmseg 1.2.2: img_scale -> scale
        # scale=(640, 640),

        flip=False,
        transforms=[
            dict(type='Resize', scale=crop_size, keep_ratio=True),  # mmseg 1.2.2: img_scale -> scale
            # dict(type='Resize', scale=(640, 640), keep_ratio=True),

            dict(type='RandomFlip',prob=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="path_to/train.list",
        pipeline=train_pipeline),
        
    val=dict(
        type=dataset_type,
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="path_to/valid.list",
        pipeline=test_pipeline),
        
    test=dict(
        type=dataset_type,
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="path_to/valid.list",
        pipeline=test_pipeline)
    )