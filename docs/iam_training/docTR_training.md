# Detection
```bash
python3 train_pytorch.py \
    /path/to/TextDetection_train \
    /path/to/TextDetection_val \
    db_resnet50 \
    --epochs 50 \
    --workers 32 \
    --pretrained \
    --freze-backbone \
    --wd 0.00001
```
| model | epochs | learning rate | weight decay|
| --- | --- | --- | --- |
| db_resnet50 | 50 | 0.001 | 0.00001 |
# Recognition