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
        zero_init_residual=False,
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')),
    cls_head=dict(
        type='I3DHead',
        num_classes=157,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        topk=3,
        loss_cls=dict(type='BCELossWithLogits')
        # transformer=True,
        # loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True),
        # label_smooth_eps=0.1
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleCharadesFrames', clip_len=64, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
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
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['mean_average_precision'])


optimizer = dict(
    type='SGD',
    lr=0.05,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001,
    # constructor='freeze_backbone_constructor',
    # paramwise_cfg = dict(lrp=0.1)
)
lr_config = dict(policy='step',
                 step=[20])

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

total_epochs = 30


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
checkpoint_config = dict(interval=5)
# load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth'
load_from = 'work_dirs/i3d_r50_64x2x1_charades_rgb/best_mean_average_precision_epoch_40.pth'
resume_from = None
work_dir = './work_dirs/i3d_r50_64x2x1_charades_rgb/'
find_unused_parameters = False

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
