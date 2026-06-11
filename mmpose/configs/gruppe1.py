_base_ = ['./_base_/default_runtime.py']

# common setting
num_keypoints = 133
input_size = (288, 384)

# runtime
max_epochs = 100
stage2_num_epochs = 10
base_lr = 5e-5
train_batch_size = 32
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=2560)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    decode_visibility=True)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/'
            'wholebody_2d_keypoint/rtmpose/ubody/rtmpose-x_simcc-ucoco_pt-aic-coco_270e-384x288-f5b50679_20230822.pth'  # noqa
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMWHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=1.,
            label_softmax=True,
            label_beta=10.,
            mask=list(range(23, 91)),
            mask_weight=0.5,
        ),
        decoder=codec),
    test_cfg=dict(flip_test=False))

# base dataset settings
dataset_type = 'CocoWholeBodyDataset'
data_mode = 'topdown'
data_root = 'data/'

backend_args = dict(backend='local')

import mmpose.utils.custom_transforms

fix_coco133 = [(i, i) for i in range(133)]

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='SafeKeypointConverter',
        num_keypoints=133,
        mapping=fix_coco133),
#    dict(type='RandomFlip', direction='horizontal'),
#    dict(type='RandomHalfBody'),
#    dict(
#        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
#    dict(type='PhotometricDistortion'),
#    dict(
#        type='Albumentation',
#        transforms=[
#            dict(type='Blur', p=0.1),
#            dict(type='MedianBlur', p=0.1),
#            dict(
#                type='CoarseDropout',
#                max_holes=1,
#                max_height=0.4,
#                max_width=0.4,
#                min_holes=1,
#                min_height=0.2,
#                min_width=0.2,
#                p=0.5),
#        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=False),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='SafeKeypointConverter',
        num_keypoints=133,
        mapping=fix_coco133),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='SafeKeypointConverter',
        num_keypoints=133,
        mapping=fix_coco133),
    #dict(type='RandomFlip', direction='horizontal'),
    #dict(type='RandomHalfBody'),
    #dict(
    #    type='RandomBBoxTransform',
    #    shift_factor=0.,
    #    scale_factor=[0.5, 1.5],
    #    rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    #dict(
    #    type='Albumentation',
    #    transforms=[
    #        dict(type='Blur', p=0.1),
    #        dict(type='MedianBlur', p=0.1),
    #    ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=False),
    dict(type='PackPoseInputs')
]

# mapping

aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
               (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]

crowdpose_coco133 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]

mpii_coco133 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_coco133 = [
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

halpe_coco133 = [(i, i)
                 for i in range(17)] + [(20, 17), (21, 20), (22, 18), (23, 21),
                                        (24, 19),
                                        (25, 22)] + [(i, i - 3)
                                                     for i in range(26, 136)]

posetrack_coco133 = [
    (0, 0),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

humanart_coco133 = [(i, i) for i in range(17)] + [(17, 99), (18, 120),
                                                  (19, 17), (20, 20)]

# train datasets
#dataset_coco = dict(
#    type=dataset_type,
#    data_root=data_root,
#    data_mode=data_mode,
#    ann_file='coco/annotations/coco_wholebody_train.json',
#    data_prefix=dict(img='coco/images'),
#    pipeline=[],
#)

hand_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[1.5, 2.0],
        rotate_factor=0),
]

interhand_left = [(21, 95), (22, 94), (23, 93), (24, 92), (25, 99), (26, 98),
                  (27, 97), (28, 96), (29, 103), (30, 102), (31, 101),
                  (32, 100), (33, 107), (34, 106), (35, 105), (36, 104),
                  (37, 111), (38, 110), (39, 109), (40, 108), (41, 91)]
interhand_right = [(i - 21, j + 21) for i, j in interhand_left]
interhand_coco133 = interhand_right + interhand_left



#dataset_hand = dict(
#    type='CombinedDataset',
#    metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody.py'),
#    datasets=[dataset_coco],
#    pipeline=[],
#    test_mode=False,
#)

#train_datasets = [dataset_hand]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        ann_file='data/coco/annotations/coco_wholebody_train.json',
        data_prefix=dict(img='data/coco/images'),
        pipeline=train_pipeline,
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        ann_file='data/coco/annotations/coco_wholebody_val.json',
        data_prefix=dict(img='data/coco/images'),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='coco-wholebody/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

import mmpose.utils.custom_metrics

# evaluators
val_evaluator = dict(
    type='SafeCocoWholeBodyMetric',
    ann_file='data/coco/annotations/coco_wholebody_val.json')
test_evaluator = val_evaluator

load_from  = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
#load_from = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
#load_from = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-x_simcc-body7_pt-body7_420e-384x288-3fbea3f2_20230731.pth"