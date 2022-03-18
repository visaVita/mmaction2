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
        drop_path_rate=0.1,
        patch_norm=True),
    cls_head=dict(
        type='TranSTHead',
        in_channels=768,
        dropout_ratio=0.5,
        num_classes=157,
        multi_class=True,
        topk=(3,5),
        tranST=dict(
            hidden_dim=768,
            enc_layer_num=0,
            stld_layer_num=2,
            n_head=6,
            dim_feedforward=768*4,
            dropout=0.0,
            drop_path_rate=0.1,
            normalize_before=False,
            fusion=True,
            rm_first_self_attn=False,
            rm_res_self_attn=True,
            activation='relu',
            return_intermediate_dec=False,
            t_only=False),
        loss_cls=dict(
            type='AsymmetricLoss',
            gamma_neg=4,
            gamma_pos=1,
            loss_weight=333.)),
    # train_cfg=dict(blending=dict(type='MixupBlending', num_classes=157, alpha=.2, smoothing=0.1)),
    test_cfg=dict(
        maximize_clips='score',
        max_testing_views=10
    ))

# dataset settings
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
        clip_len=32,
        frame_interval=2,
        num_clips=1),
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
        type='SampleCharadesFrames',
        clip_len=32,
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
        clip_len=32,
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
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=2),
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
optimizer = dict(type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(
                 custom_keys=dict(
                     # swin param
                     absolute_pos_embed=dict(decay_mult=0.0),
                     relative_position_bias_table=dict(decay_mult=0.0),
                     norm=dict(decay_mult=0.0),
                     # TranSTL param
                     # pos_enc=dict(lr_mult=0.0, decay_mult=0.0),
                     # bias=dict(decay_mult=0.0),
                     # TranST=dict(lr_mult=0.5),
                     # lrp param
                     backbone=dict(lr_mult=0.1)))
)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 30

# runtime settings
work_dir = './work_dirs/k400_swin_tiny_1k_patch244_window877_tranST_charades'
find_unused_parameters = True
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://github.com/SwinTransformer/storage/releases/download/'
             'v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth')
# load_from = ('work_dirs/k400_swin_tiny_1k_patch244_window877_charades/map4210.pth')
# load_from = None
resume_from = None
workflow = [('train', 1)]
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)