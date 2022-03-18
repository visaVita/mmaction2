model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        depths=[2, 2, 18, 2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True),
    cls_head=dict(
        type='TranSTHead',
        in_channels=1024,
        dropout_ratio=0.5,
        num_classes=157,
        multi_class=True,
        label_smooth_eps=0.1,
        topk=(3, 5),
        tranST=dict(hidden_dim=1024,
                    enc_layer_num=0,
                    stld_layer_num=2,
                    n_head=8,
                    dim_feedforward=4096,
                    dropout=0.,
                    drop_path_rate=0.1,
                    normalize_before=False,
                    fusion=False,
                    rm_first_self_attn=True,
                    rm_res_self_attn=True,
                    activation="relu",
                    return_intermediate_dec=False,
                    t_only=False
        ),
        loss_cls=dict(
            type='AsymmetricLoss',
            gamma_neg=4,
            gamma_pos=1,
            # disable_torch_grad_focal_loss=True
        )
    ),
    # train_cfg=dict(blending=dict(type='MixupBlending', num_classes=157, alpha=.2, smoothing=0.1)),
    test_cfg=dict(maximize_clips='score', max_testing_views=5))

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
        clip_len=32,
        frame_interval=2,
        num_clips=1),
    # dict(type='UniformSampleCharadesFrames', clip_len=64),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 340)),
    dict(type='RandomCrop', size=224),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=32,
        num_clips=1,
        frame_interval=2,
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
        num_clips=10,
        frame_interval=2,
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

evaluation = dict(interval=1, metrics=['mean_average_precision'])
optimizer = dict(
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    # eps=1e-8,
    weight_decay=0.05,
    amsgrad=False,
    # constructor='swin_constructor',
    paramwise_cfg=dict(
        custom_keys={
            # swin param
            'backbone.absolute_pos_embed': dict(decay_mult=0.0),
            'backbone.relative_position_bias_table': dict(decay_mult=0.0),
            'backbone.norm': dict(decay_mult=0.0),
            # lrp param
            'backbone': dict(lr_mult=0.1),
            # 'cls_head': dict(decay_mult=0.2)
        }
    )
)

# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=dict(max_norm=5., norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)

total_epochs = 30

work_dir = './work_dirs/k600_swin_base_22k_patch244_window877_tranST_charades'
find_unused_parameters = False
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth')
resume_from = None
workflow = [('train', 1)]
omnisource = False