model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=65,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        transformer=True,
        loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True)),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(maximize_clips='score'))

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
        num_clips=10,
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
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['mean_average_precision'])

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook'),
    ])

""" optimizer = dict(
    type='SGD',
    lr=0.005,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001) """
optimizer = dict(type='AdamW',
                 lr=2e-5,
                 betas=(0.9, 0.9999),
                 weight_decay=2e-2,
                 paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                                   'bn': dict(decay_mult=0.)})
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[15, 25])

""" lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=4) """
lr_config = dict(
    policy='CosineRestart',
    periods=[20, 40, 60],
    restart_weights=[1, 0.8, 0.6],
    min_lr = 0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2
)

total_epochs = 80

# runtime settings
checkpoint_config = dict(interval=10)
# load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth'
load_from = '/home/ckai/project/mmaction2/work_dirs/i3d_r50_32x2x1_120e_multithumos_rgb/map7990_68e_AdamW1e-2_asl_4_1.pth'
resume_from = None
work_dir = './work_dirs/i3d_r50_32x2x1_120e_multithumos_rgb/'
find_unused_parameters = True

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
