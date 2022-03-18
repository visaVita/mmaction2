_base_ = '../../_base_/models/movinet/movinetA5.py'

model = dict(
    cls_head=dict(
        type='MoViNetHead',
        in_channels=992,
        hidden_dim=2048,
        num_classes=157,
        spatial_type='avg',
        tf_like=True,
        causal=False,
        conv_type='3d',
        dropout_ratio=0.,
        # label_smooth_eps=0.1,
        topk=(3),
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits')
        # loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True)
    ),
    train_cfg=None,
    test_cfg=dict(maximize_clips='score')
)

dataset_type = 'CharadesDataset'
data_root = 'data/charades/Charades_rgb'
data_root_val = 'data/charades/Charades_rgb'
ann_file_train = 'data/charades/annotations/charades_train_list_rawframes.csv'
ann_file_val = 'data/charades/annotations/charades_val_list_rawframes.csv'
ann_file_test = 'data/charades/annotations/charades_val_list_rawframes.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=64,
        frame_interval=2,
        num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 340)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleCharadesFrames',
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
    videos_per_gpu=1,
    workers_per_gpu=2,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(
    interval=1, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(type='AdamW',
                 lr=0.0001,
                 betas=(0.9, 0.9999),
                 weight_decay=0.05,
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# do not use mmdet version fp16
# fp16 = dict()
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=4,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=4
)
total_epochs = 48

# runtime settings
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
log_level = 'INFO'
work_dir = './work_dirs/movinetA5_charades/'

load_from = ('model_zoo/movinet/movinet_A5.pth')
# load_from = None
find_unused_parameters = False
resume_from = None
dist_params = dict(backend='nccl')