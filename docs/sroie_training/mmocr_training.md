
# Detection
This has to be saved inside `configs/textdet/_base_/datasets` and I called it `sroie.py`
```python
sroie_textdet_data_root = "data/sroie"
sroie_textdet_train = dict(
    type="OCRDataset",
    data_root=sroie_textdet_data_root,
    ann_file="textdet_train.json",
    pipeline=None
)
sroie_textdet_test = dict(
    type="OCRDataset",
    data_root=sroie_textdet_data_root,
    ann_file="textdet_val.json",
    test_mode=True,
    pipeline=None
)
```
This has to be saved inside `configs/textdet/dbnet` and I called it `dbnet_resnet50-dcnv2_fpnc_1200e_sroie.py`
```python
_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/sroie.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]
# dataset settings
sroie_textdet_train = _base_.sroie_textdet_train
sroie_textdet_train.pipeline = _base_.train_pipeline
sroie_textdet_test = _base_.sroie_textdet_test
sroie_textdet_test.pipeline = _base_.test_pipeline
train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=sroie_textdet_train)
val_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=sroie_textdet_test)
test_dataloader = val_dataloader
auto_scale_lr = dict(base_batch_size=12)
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
The full config file is [here](/docs/sroie_training/config/dbnet_resnet50-dcnv2_fpnc_1200e_sroie.py)
| model | epochs | optimizer | learning rate |
| --- | --- | --- | --- |
| DBNet | 1200 | SGD | 0.007 |
# Recognition
This has to be saved inside `configs/textrecog/_base_/datasets` and I called it `sroie.py`
```python
sroie_textrecog_data_root = "data/sroie"
sroie_textrecog_train = dict(
    type="OCRDataset",
    data_root=sroie_textrecog_data_root,
    ann_file="textrecog_train.json",
    pipeline=None
)
sroie_textrecog_test = dict(
    type="OCRDataset",
    data_root=sroie_textrecog_data_root,
    ann_file="textrecog_val.json",
    test_mode=True,
    pipeline=None
)
```
This has to be saved inside `configs/textrecog/aster` and I called it `aster_resnet45_6e_st_mj_sroie.py`
```python
_base_ = [
    '_base_aster.py',
    '../_base_/datasets/sroie.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_6e.py',
]
# dataset settings
train_list = [_base_.sroie_textrecog_train]
test_list = [_base_.sroie_textrecog_test]
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)
train_dataloader = dict(
    batch_size=1024,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
auto_scale_lr = dict(base_batch_size=1024)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)
val_dataloader = test_dataloader
val_evaluator = dict(dataset_prefixes=['sroie'])
test_evaluator = val_evaluator
```
The full config file is [here](/docs/sroie_training/config/aster_resnet45_6e_st_mj_sroie.py)
| model | epochs | optimizer | learning rate |
| --- | --- | --- | --- |
| ASTER | 500 | AdamW | 0.0004 |