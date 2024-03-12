# Detection
To train detection easyOCR already provides a sample configuration inside `trainer/craft/config/`, this is the [configuration](/docs/sroie_training/config/sroie_data_train.yaml) that it's been used
This is the command to start training:
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=sroie_data_train
```
| model | epochs | learning rate | weight decay |
| --- | --- | --- | --- |
| CRAFT_clr_amp_29500 | 1200 | 0.0001 | 0.00001 |
# Recognition