# Detection
To train detection easyOCR already provides a sample configuration inside `trainer/craft/config/`, this is the [configuration](/docs/funsd_training/config/funsd_data_train.yaml) that it's been used
This is the command to start training:
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=funsd_data_train
```
| model | epochs | learning rate | weight decay |
| --- | --- | --- | --- |
| CRAFT_clr_amp_29500 | 1200 | 0.0001 | 0.00001 |
# Recognition
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --train_data result/funsd/train/
    --valid_data result/funsd/val/
    --select_data "/" 
    --batch_ratio 1.0 
    --Transformation TPS 
    --FeatureExtraction ResNet 
    --SequenceModeling BiLSTM 
    --Prediction Attn 
    --batch_size 48
    --data_filtering_off
    --workers 16
    --batch_max_length 50
    --num_iter 1200 
    --valInterval 5 
    --saved_model TPS-ResNet-BiLSTM-Attn.pth
```
| model | epochs | learning rate | weight decay |
| --- | --- | --- | --- |
| TPS-ResNet-BiLSTM-Attn | 1200 | 1 | 0.00001 |
