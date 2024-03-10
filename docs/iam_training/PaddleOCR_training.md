# Detection
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
           tools/train.py -c configs/det/det_mv3_db.yml \
                          -o Global.use_gpu=true \
                             Global.save_model_dir="./output/iam-det_mv3_db/" \
                             Train.dataset.data_dir=./train_data/iam/Detection/train_img \
                             Train.dataset.label_file_list=["./train_data/iam/Detection/train_label.txt"] \
                             Train.loader.num_workers=32 \
                             Eval.dataset.data_dir=./train_data/iam/Detection/val_img \
                             Eval.dataset.label_file_list=["./train_data/iam/Detection/val_label.txt"] \
                             Eval.loader.num_workers=32
```
The full config file is [here](/docs/iam_training/config/iam-det_mv3_db.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| MobileNetV3_large_x0_5_pretrained | 1200 | Adam | 0.001 | L2 |
# Recognition
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
           tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
                          -o Global.use_gpu=true \
                             Global.pretrained_model="./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy" \
                             Global.save_model_dir="./output/iam-en_PP-OCRv3_rec/" \
                             Train.dataset.data_dir=./train_data/iam/Recognition/train_img \
                             Train.dataset.label_file_list=["./train_data/iam/Recognition/train_label.txt"] \
                             Train.loader.num_workers=64 \
                             Eval.dataset.data_dir=./train_data/iam/Recognition/val_img \
                             Eval.dataset.label_file_list=["./train_data/iam/Recognition/val_label.txt"] \
                             Eval.loader.num_workers=64
```
The full config file is [here](/docs/iam_training/config/iam-en_PP-OCRv3_rec.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| en_PP-OCRv3_rec_train | 500 | Adam | 0.001 | L2 |