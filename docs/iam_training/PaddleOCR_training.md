# Detection
The model chosen for training is `MobileNetV3_large_x0_5_pretrained`, the number of epochs are 1200 and the command used to start the training is shown below:
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
| epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- |
| 1200 | Adam | 0.001 | L2 |
# Recognition
The model chosen for training is `en_PP-OCRv3_rec_train`, the command used to start the training is shown below:
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
| epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- |
| 500 | Adam | 0.001 | L2 |