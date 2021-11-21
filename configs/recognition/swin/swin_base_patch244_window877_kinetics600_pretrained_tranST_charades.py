model=dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2,4,4),
        depths=[2, 2, 18, 2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True),
    cls_head=dict(
        type='TranSTHead',
        in_channels=1024,
        spatial_type='avg',
        dropout_ratio=0.5,
        num_classes=157,
        multi_class=True,
        topk=(3),
        tranST=dict(hidden_dim=1024,
                    enc_layer_num=1,
                    stld_layer_num=4,
                    n_head=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    normalize_before=True,
                    fusion="t2s",
                    rm_self_attn_dec=False,
                    rm_first_self_attn=False,
                    activation="gelu",
                    return_intermediate_dec=False,
                    t_only=False
        ),
        label_smooth_eps=0.1,
        # loss_cls=dict(type='BCELossWithLogits'),
        loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True),
        ),
    # train_cfg=dict(blending=dict(type='LabelSmoothing', num_classes=157, smoothing=0.1)),
    test_cfg=dict(
        maximize_clips='score',
        max_testing_views=2
    ))

# dataset settings
dataset_type = 'CharadesDataset'
data_root = 'data/charades/Charades_rgb'
data_root_val = 'data/charades/Charades_rgb'
ann_file_train = 'data/charades/annotations/charades_train_list_rawframes.csv'
ann_file_val = 'data/charades/annotations/charades_val_list_rawframes.csv'
ann_file_test = 'data/charades/annotations/charades_val_list_rawframes.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    # mean=[105.315, 93.84, 86.19], std=[33.405, 31.875, 33.66],
    to_bgr=False)
train_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=64,
        frame_interval=2,
        num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 340)),
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
        type='SampleCharadesFrames',
        clip_len=64,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
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
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
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
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['mean_average_precision'])

# optimizer

optimizer = dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 #'pos_s': dict(decay_mult=0.),
                                                 #'pos_t': dict(decay_mult=0.),
                                                 #'norm1': dict(decay_mult=0.),
                                                 #'norm2': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))

# optimizer = dict(type='SGD',
#                  lr=5e-2,
#                  weight_decay=1e-4,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.),
#                                                  'backbone': dict(lr_mult=0.1)})
#                  )
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))


# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)

# lr_config = dict(
#     policy='OneCycle',
#     max_lr=75e-6)

total_epochs = 30

# runtime settings
work_dir = './work_dirs/k400_swin_base_22k_patch244_window877_charades'
find_unused_parameters = True
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
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
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = ('https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth')
# load_from = ('work_dirs/k400_swin_base_22k_patch244_window877/best_mean_average_precision_epoch_35.pth')
# load_from = None
resume_from = None
workflow = [('train', 1)]
# custom_hooks=[dict(type='EMAHook', momentum=0.0003)]
