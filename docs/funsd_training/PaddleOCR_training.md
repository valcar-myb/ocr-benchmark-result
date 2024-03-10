# Detection
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
        tools/train.py -c configs/det/det_mv3_db.yml \
                       -o Global.use_gpu=true \
                          Global.save_model_dir="./output/funsd-det_mv3_db/" \
                          Global.eval_batch_step=50 \
                          Train.dataset.data_dir=./train_data/funsd/Detection/train_img \
                          Train.dataset.label_file_list=["./train_data/funsd/Detection/train_label.txt"] \
                          Train.loader.num_workers=32 \
                          Eval.dataset.data_dir=./train_data/funsd/Detection/val_img \
                          Eval.dataset.label_file_list=["./train_data/funsd/Detection/val_label.txt"] \
                          Eval.loader.num_workers=32
```
The full config file is [here](/docs/funsd_training/config/funsd-det_mv3_db.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| MobileNetV3_large_x0_5_pretrained | 1200 | Adam | 0.001 | L2 |
# Recognition
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
        tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
                       -o Global.use_gpu=true \
                          Global.save_epoch_step=500 \
                          Global.eval_batch_step=100 \
                          Global.pretrained_model="./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy" \
                          Global.save_model_dir="./output/funsd-en_PP-OCRv3_rec/" \
                          Train.dataset.data_dir=./train_data/funsd/Recognition/train_img \
                          Train.dataset.label_file_list=["./train_data/funsd/Recognition/train_label.txt"] \
                          Train.loader.num_workers=64 \
                          Eval.dataset.data_dir=./train_data/funsd/Recognition/val_img \
                          Eval.dataset.label_file_list=["./train_data/funsd/Recognition/val_label.txt"] \
                          Eval.loader.num_workers=64
```
The full config file is [here](/docs/funsd_training/config/funsd-en_PP-OCRv3_rec.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| en_PP-OCRv3_rec_train | 500 | Adam | 0.001 | L2 |