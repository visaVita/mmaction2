_base_ = '../../_base_/models/movinet/movinetA4.py'

model = dict(
    cls_head=dict(
        num_classes=157,
        multi_class=True,
        loss_cls=dict(type='AsymmetricLossOptimized', gamma_neg=1, gamma_pos=0),
        dropout_ratio=0.2,
        topk=(1,3,5),
        label_smooth_eps=0.1
    ),
    train_cfg=None,
    test_cfg=dict(maximize_clips='score')
)

dataset_type = 'CharadesDataset'
data_root = 'data/charades/rawframes'
data_root_val = 'data/charades/rawframes'
ann_file_train = 'data/charades/annotations/charades_train_list_rawframes.csv'
ann_file_val = 'data/charades/annotations/charades_val_list_rawframes.csv'
ann_file_test = 'data/charades/annotations/charades_val_list_rawframes.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(
        type='SampleCharadesFrames',
        clip_len=64,
        frame_interval=4,
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
        clip_len=64,
        frame_interval=4,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
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
        frame_interval=4,
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
    interval=5, metrics=['mean_average_precision'])

# optimizer
""" optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
    constructor='transformer_mlc_optimizer_constructor',
    paramwise_cfg = dict(lrp=0.1)
) """
optimizer = dict(type='AdamW',
                 lr=0.0001,
                 betas=(0.9, 0.9999),
                 weight_decay=0.01,
                 # constructor='transformer_mlc_optimizer_constructor',
                 # paramwise_cfg = dict(lrp=0.1)
                 # paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[15, 25],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2
    )
""" lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34) """
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
log_level = 'INFO'
work_dir = './work_dirs/movinetA4/'

load_from = ('/home/ckai/project/mmaction2/model_zoo/movinet/modelA4_statedict_mm')
# load_from = ('work_dirs/movinetA4/best_mean_average_precision_epoch_38.pth')
# load_from = None
find_unused_parameters = True
resume_from = None
dist_params = dict(backend='nccl')
