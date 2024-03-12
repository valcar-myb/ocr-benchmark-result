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
    --wd 0.00001 \
    -b 8
```
| model | epochs | learning rate | weight decay|
| --- | --- | --- | --- |
| db_resnet50 | 50 | 0.001 | 0.00001 |
# Recognition
```bash
python3 train_pytorch.py \
    crnn_vgg16_bn \
    --train_path /path/to/TextRecognition_train \
    --val_path /path/to/TextRecognition_val \
    --epochs 50 \
    --vocab sroie
```
The sroie dataset has some character that are not in the default vocab of doctr, to solve it just add this line of code in `/doctr/doctr/datasets/vocabs.py`:
```python
VOCABS["sroie"] = VOCABS["french"] + " ·"
```
| model | epochs | learning rate |
| --- | --- | --- |
| crnn_vgg16_bn | 50 | 0.001 |