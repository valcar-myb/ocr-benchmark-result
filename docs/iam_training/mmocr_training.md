
# Detection
This has to be saved inside `configs/textdet/_base_/datasets` and I called it `iam.py`
```python
iam_textdet_data_root = "data/iam"
iam_textdet_train = dict(
    type="OCRDataset",
    data_root=iam_textdet_data_root,
    ann_file="textdet_train.json",
    pipeline=None
)
iam_textdet_test = dict(
    type="OCRDataset",
    data_root=iam_textdet_data_root,
    ann_file="textdet_val.json",
    test_mode=True,
    pipeline=None
)
```
This has to be saved inside `configs/textdet/dbnet` and I called it `dbnet_resnet50-dcnv2_fpnc_1200e_iam.py`
```python
_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/iam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]
# dataset settings
iam_textdet_train = _base_.iam_textdet_train
iam_textdet_train.pipeline = _base_.train_pipeline
iam_textdet_test = _base_.iam_textdet_test
iam_textdet_test.pipeline = _base_.test_pipeline
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=iam_textdet_train
)
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=iam_textdet_test
    )
test_dataloader = val_dataloader
auto_scale_lr = dict(base_batch_size=16)
```
This is the `schedule_sgd_1200e.py` where I modified the number of epochs
```python
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1200),
]
```
The full config file is [here](/docs/iam_training/config/dbnet_resnet50-dcnv2_fpnc_1200e_iam.py)
| model | epochs | optimizer | learning rate |
| --- | --- | --- | --- |
| DBNet | 1200 | SGD | 0.007 |
# Recognition
This has to be saved inside `configs/textdet/_base_/datasets` and I called it `iam.py`
```python
iam_textrecog_data_root = "data/iam"
iam_textrecog_train = dict(
    type="OCRDataset",
    data_root=iam_textrecog_data_root,
    ann_file="textrecog_train.json",
    pipeline=None
)
iam_textrecog_test = dict(
    type="OCRDataset",
    data_root=iam_textrecog_data_root,
    ann_file="textrecog_val.json",
    test_mode=True,
    pipeline=None
)
```
This has to be saved inside `configs/textdet/crnn` and I called it `crnn_mini-vgg_5e_iam.py`
```python
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/iam.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]
# dataset settings
train_list = [_base_.iam_textrecog_train]
test_list = [_base_.iam_textrecog_test]
default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader
_base_.model.decoder.dictionary.update(
    dict(with_unknown=True, unknown_token=None))
_base_.train_cfg.update(dict(max_epochs=500, val_interval=5))
val_evaluator = dict(dataset_prefixes=['iam'])
test_evaluator = val_evaluator
```
The full config file is [here](/docs/iam_training/config/crnn_mini-vgg_5e_iam.py)
| model | epochs | optimizer | learning rate |
| --- | --- | --- | --- |
| CRNN | 500 | Adadelta | 1.0 |