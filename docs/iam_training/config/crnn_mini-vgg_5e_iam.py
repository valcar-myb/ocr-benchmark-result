default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dictionary = dict(
    dict_file=
    '/mnt/sdb1/workspace/mybiros/mmocr_api/mmocr/configs/textrecog/crnn/../../../dicts/lower_english_digits.txt',
    type='Dictionary',
    with_padding=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
iam_textrecog_data_root = 'data/iam'
iam_textrecog_test = dict(
    ann_file='textrecog_val.json',
    data_root='data/iam',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
iam_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='data/iam',
    pipeline=None,
    type='OCRDataset')
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(input_channels=1, leaky_relu=False, type='MiniVGG'),
    data_preprocessor=dict(
        mean=[
            127,
        ], std=[
            127,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file=
            '/mnt/sdb1/workspace/mybiros/mmocr_api/mmocr/configs/textrecog/crnn/../../../dicts/lower_english_digits.txt',
            type='Dictionary',
            unknown_token=None,
            with_padding=True,
            with_unknown=True),
        in_channels=512,
        module_loss=dict(letter_case='lower', type='CTCModuleLoss'),
        postprocessor=dict(type='CTCPostProcessor'),
        rnn_flag=True,
        type='CRNNDecoder'),
    encoder=None,
    preprocessor=None,
    type='CRNN')
optim_wrapper = dict(
    optimizer=dict(lr=1.0, type='Adadelta'), type='OptimWrapper')
param_scheduler = [
    dict(factor=1.0, type='ConstantLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_val.json',
                data_root='data/iam',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(color_type='grayscale', type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=None,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_prefixes=[
        'iam',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='textrecog_val.json',
        data_root='data/iam',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(color_type='grayscale', type='LoadImageFromFile'),
    dict(
        height=32,
        max_width=None,
        min_width=32,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=500, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='data/iam',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='grayscale',
                ignore_empty=True,
                min_size=2,
                type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(keep_ratio=False, scale=(
                100,
                32,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='data/iam',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(
        color_type='grayscale',
        ignore_empty=True,
        min_size=2,
        type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(keep_ratio=False, scale=(
        100,
        32,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(color_type='grayscale', type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(
                    height=32,
                    max_width=None,
                    min_width=32,
                    type='RescaleToHeight',
                    width_divisor=16),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_val.json',
                data_root='data/iam',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(color_type='grayscale', type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=None,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'iam',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/crnn_mini-vgg_5e_iam'