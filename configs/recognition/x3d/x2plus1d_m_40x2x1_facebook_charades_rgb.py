# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='X3D',
        pretrained=None,
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2,
        se_ratio=None,
        use_swish=False,
        bn_frozen=False,
        frozen_stages=-1,
        conv_cfg=(
            # dict(type='Conv3d'),
            dict(type='Conv2plus1d'),
            dict(type='Conv2plus1d'),
            dict(type='Conv2plus1d'),
            dict(type='Conv2plus1d'),
            # dict(type='Conv2plus1d')
        )
    ),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=157,
        spatial_type='avg',
        dropout_ratio=0.5,
        topk=(3),
        multi_class=True,
        # loss_cls=dict(type='BCELossWithLogits'),
        loss_cls=dict(
            type='AsymmetricLossOptimized',
            gamma_neg=4,
            gamma_pos=1,
            disable_torch_grad_focal_loss=True),
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(maximize_clips='score'))

# dataset settings
dataset_type = 'CharadesDataset'
data_root = 'data/charades/Charades_rgb'
data_root_val = 'data/charades/Charades_rgb'
ann_file_train = 'data/charades/annotations/charades_train_list_rawframes.csv'
ann_file_val = 'data/charades/annotations/charades_val_list_rawframes.csv'
ann_file_test = 'data/charades/annotations/charades_val_list_rawframes.csv'
img_norm_cfg = dict(
    mean=[105.315, 93.84, 86.19], std=[33.405, 31.875, 33.66], to_bgr=False)
train_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=40,
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
        clip_len=40,
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
        clip_len=40,
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
    videos_per_gpu=4,
    workers_per_gpu=2,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=4, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                #  type='SGD', lr=0.05, momentum=0.9, weight_decay=1e-04,
                #  paramwise_cfg=dict(
                #     custom_keys={
                #         'backbone.conv1': dict(lr_mult=2.0),
                #         'backbone.layer4': dict(lr_mult=2.0),
                #     }
                #  )
)


# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=4,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=17)
total_epochs = 196

# runtime settings
work_dir = './work_dirs/x2plus1d_charades'
find_unused_parameters = False
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/x2plus1d_charades/best_mean_average_precision_epoch_56.pth'
# load_from = 'model_zoo/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
resume_from = None
workflow = [('train', 1)]