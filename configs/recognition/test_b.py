model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNetTSM',
        depth=50,
        num_segments=8,
        is_shift=True,
        shift_div=8,
        shift_place='blockres',
        temporal_pool=False,),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=157,
        multi_class=True,
        spatial_type='avg',
        loss_cls=dict(type='BCELossWithLogits', loss_weight=80.),
        dropout_ratio=0.5),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
    
dataset_type = 'RawframeDataset'
data_root = 'data/Charades/Charades_v1_rgb'
data_root_val = 'data/Charades/Charades_v1_rgb'
ann_file_train = 'data/Charades/train.txt'
ann_file_val = 'data/Charades/test.txt'
ann_file_test = 'data/Charades/test.txt'
img_norm_cfg = dict(
    mean=[105.315, 93.84, 86.19], std=[33.405, 31.875, 33.66], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=10, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=10,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=10,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=157,
        filename_tmpl='img_{:05}.jpg'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=157,
        filename_tmpl='img_{:05}.jpg'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=157,
        filename_tmpl='img_{:05}.jpg'))
evaluation = dict(
    interval=5, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.0375, momentum=0.9,
    weight_decay=1e-4)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-3,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
total_epochs = 120

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
log_level = 'INFO'
work_dir = './work_dirs/test_bb'
load_from = None
#             'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
#             'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')
find_unused_parameters = False
resume_from = None
workflow = [('train', 1)]
