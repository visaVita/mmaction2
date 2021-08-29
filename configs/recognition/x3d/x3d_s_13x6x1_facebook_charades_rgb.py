_base_ = ['../../_base_/models/x3d.py']

model = dict(
    cls_head=dict(
        num_classes=157,
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits')
    ),
    train_cfg=None,
    test_cfg=dict(maximize_clips='score'))

# dataset settings
dataset_type = 'CharadesDataset'
data_root = 'data/charades/rawframes'
data_root_val = 'data/charades/rawframes'
ann_file_train = 'data/charades/annotations/charades_train_list_rawframes.csv'
ann_file_val = 'data/charades/annotations/charades_val_list_rawframes.csv'
ann_file_test = 'data/charades/annotations/charades_val_list_rawframes.csv'
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=6,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 192)),
    dict(type='CenterCrop', crop_size=192),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

dist_params = dict(backend='nccl')
