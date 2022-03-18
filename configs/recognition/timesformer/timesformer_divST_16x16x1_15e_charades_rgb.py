_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained= None, # noqa: E251
        # 'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=16,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=157,
        in_channels=768,
        topk=(3),
        multi_class=True,
        loss_cls=dict(
            type='AsymmetricLossOptimized',
            gamma_neg=4,
            gamma_pos=1,
            disable_torch_grad_focal_loss=True),
    ),
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
    dict(type='SampleCharadesFrames', clip_len=16, frame_interval=16, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
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
        clip_len=16,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=16,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 30])
total_epochs = 40
load_from = "https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth"
# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/timesformer_divST_16x16x1_40e_charades_rgb'
