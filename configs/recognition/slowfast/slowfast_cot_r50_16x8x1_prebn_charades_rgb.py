model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False,
            CoT=(0, 0, 0, 0),
            frozen_stages=-1
            ),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False,
            CoT=(0, 0, 0, 0),
            frozen_stages=-1
            )),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=157,
        multi_class=True,
        spatial_type='avg',
        # loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=4, gamma_pos=1, disable_torch_grad_focal_loss=True),
        transformer=False,
        loss_cls=dict(type='BCELossWithLogits'),
        dropout_ratio=0.5,
        # label_smooth_eps=0.1
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(maximize_clips='score'))
    
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
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(
    interval=6, metrics=['mean_average_precision'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.15,
    momentum=0.9,
    weight_decay=0.0001,
    # constructor='transformer_mlc_optimizer_constructor',
    # paramwise_cfg = dict(lrp=0.1),
    # paramwise_cfg = dict(paramwise_cfg = dict(custom_keys={
    #                 'backbone': dict(lr_mult=1., decay_mult=1.),
    #                 'cls_head': dict(lr_mult=0.1, decay_mult=10.),
    #                 }),)
)
# optimizer = dict(type='AdamW',
#                  lr=4e-5,
#                  betas=(0.9, 0.9999),
#                  weight_decay=2e-2,
#                  paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1),
#                                                    'bn': dict(decay_mult=0.)})
# )

optimizer_config = dict(
    grad_clip=dict(
        max_norm=40,
        norm_type=2)
)
# learning policy
lr_config = dict(
    policy='step',
    step=[41, 49],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=4,
    min_lr=1e-4,
)

""" lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=4) """

# lr_config = dict(
#     policy='CosineRestart',
#     periods=[20, 40, 60],
#     restart_weights=[1, 1, 1],
#     min_lr = 0,
#     warmup='linear',
#     warmup_by_epoch=True,
#     warmup_iters=4
# )
# precise_bn
precise_bn = dict(num_iters=200, interval=6)

total_epochs = 57

# runtime settings
checkpoint_config = dict(interval=6)
workflow = [('train', 1)]
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
log_level = 'INFO'
work_dir = './work_dirs/slowfast_cot_r50_16x8x1_prebn_charades_rgb'

load_from = ('https://download.openmmlab.com/mmaction/recognition/'
                 'slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/'
                 'slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth')
# load_from = ('/home/ckai/project/mmaction2/work_dirs/slowfast_cot_r50_8x8x1_57e_charades_rgb/map3642_lr_1875_asl_4_1_58e.pth')
# load_from = None
find_unused_parameters = False
# recompute_scale_factor = True
resume_from = None
dist_params = dict(backend='nccl')
