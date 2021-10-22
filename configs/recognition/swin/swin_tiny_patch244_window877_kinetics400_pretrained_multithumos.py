model=dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2,4,4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        spatial_type='avg',
        dropout_ratio=0.5,
        num_classes=65,
        multi_class=True,
        loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True),
        ),
    test_cfg=dict(
        maximize_clips='score',
        max_testing_views=10
    ))

# dataset settings
dataset_type = 'MultiThumosDataset'
data_root = 'data/multithumos/rawframes'
data_root_val = 'data/multithumos/rawframes'
ann_file_train = 'data/multithumos/annotations/val.csv'
ann_file_val = 'data/multithumos/annotations/test.csv'
ann_file_test = 'data/multithumos/annotations/test.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(
        type='SampleCharadesFrames',
        num_classes=65,
        clip_len=64,
        frame_interval=2,
        num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 340)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleCharadesFrames',
        num_classes=65,
        clip_len=64,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleCharadesFrames',
        num_classes=65,
        clip_len=64,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']) 
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['mean_average_precision'])

# optimizer
""" optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)})) """
optimizer = dict(type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
""" lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
) """
lr_config = dict(
    policy='CosineRestart',
    periods=[20, 40, 60],
    restart_weights=[1, 0.8, 0.6],
    min_lr = 0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2
)
""" lr_config = dict(
    policy='step',
    step=[20],
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001) """
total_epochs = 80

# runtime settings
work_dir = './work_dirs/k400_swin_base_22k_patch244_window877_multithumos'
find_unused_parameters = False
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://github.com/SwinTransformer/storage/releases/download/'
             'v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth')
# load_from = ('https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth')
# load_from = None
resume_from = '/home/ckai/project/mmaction2/work_dirs/k400_swin_base_22k_patch244_window877_multithumos/epoch_10.pth'
# resume_from = '/home/ckai/project/mmaction2/work_dirs/k400_swin_base_22k_patch244_window877/epoch_9.pth'
workflow = [('train', 1)]
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)