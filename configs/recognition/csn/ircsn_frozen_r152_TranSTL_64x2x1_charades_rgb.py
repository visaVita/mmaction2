_base_ = [
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained=None,
        depth=152,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        frozen_stages=-1,
        # pretrained=  noqa: E251
        # 'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'  # noqa: E501
        zero_init_residual=False),
    cls_head=dict(
        type='TranSTHead',
        in_channels=2048,
        dropout_ratio=0.5,
        num_classes=157,
        multi_class=True,
        # label_smooth_eps=0.1,
        topk=3,
        tranST=dict(hidden_dim=2048,
                    enc_layer_num=0,
                    stld_layer_num=2,
                    n_head=4,
                    dim_feedforward=2048*4,
                    dropout=0.,
                    drop_path_rate=0.2,
                    normalize_before=False,
                    fusion=True,
                    rm_first_self_attn=False,
                    rm_res_self_attn=True,
                    activation="relu",
                    return_intermediate_dec=False,
                    t_only=False
        ),
        loss_cls=dict(
            type='AsymmetricLoss',
            gamma_neg=4,
            gamma_pos=1)
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(
        maximize_clips='score',
        # max_testing_views=5
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
    # dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
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
evaluation = dict(interval=1, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)  # this lr is used for 8 gpus
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=4
)
total_epochs = 60

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
find_unused_parameters = True
load_from = 'model_zoo/ircsn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'
work_dir = './work_dirs/ircsn_r152_TranSTL_32x2x1_charades_rgb'  # noqa: E501
