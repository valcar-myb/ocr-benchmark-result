# Detection
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
        tools/train.py -c configs/det/det_r50_vd_db.yml \
                       -o Global.use_gpu=true \
                          Global.save_model_dir="./output/sroie-det_r50_vd_db/" \
                          Train.dataset.data_dir=./train_data/sroie/Detection/train_img \
                          Train.dataset.label_file_list=["./train_data/sroie/Detection/train_label.txt"] \
                          Train.loader.batch_size_per_card=8 \
                          Eval.dataset.data_dir=./train_data/sroie/Detection/val_img \
                          Eval.dataset.label_file_list=["./train_data/sroie/Detection/val_label.txt"]
```
The full config file is [here](/docs/sroie_training/config/sroie-det_r50_vd_db.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| ResNet50_vd_ssld_pretrained.pdparams | 1200 | Adam | 0.001 | L2 |
# Recognition
```bash
python3 -m paddle.distributed.launch --gpus '0,1,2,3' \
        tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml \
                       -o Global.use_gpu=true \
                          Global.save_epoch_step=500 \
                          Global.eval_batch_step=200 \
                          Global.pretrained_model="./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy" \
                          Global.save_model_dir="./output/sroie-en_PP-OCRv3_rec/" \
                          Train.dataset.data_dir=./train_data/sroie/Recognition/train_img \
                          Train.dataset.label_file_list=["./train_data/sroie/Recognition/train_label.txt"] \
                          Train.loader.num_workers=64 \
                          Eval.dataset.data_dir=./train_data/sroie/Recognition/val_img \
                          Eval.dataset.label_file_list=["./train_data/sroie/Recognition/val_label.txt"] \
                          Eval.loader.num_workers=64
```
The full config file is [here](/docs/sroie_training/config/sroie-en_PP-OCRv3_rec.yml)
| model | epochs | optimizer | learning rate | regularizer |
| --- | --- | --- | --- | --- |
| en_PP-OCRv3_rec_train | 500 | Adam | 0.001 | L2 |