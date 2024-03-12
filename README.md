# ocr-benchmark-result
ocr-benchmark-result
# models
Before training/finetuning (for every dataset)

| | Detection model | Recognition model |
| --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` |
| paddleOCR | `en_PP-OCRv3_det_infer` | `en_PP-OCRv4_rec_infer` |
| mmOCR | `DBNet` | `CRNN` |
| easyOCR | `craft` | `standard` |
| tesseract | . | . |

# training
Below are the models used for training the various datasets.
## training FUNSD dataset
| | Detection model | Recognition model | train-config |
| --- | --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` | [config](/docs/funsd_training/docTR_training.md) |
| paddleOCR | `MobileNetV3_large_x0_5_pretrained` | `en_PP-OCRv3_rec_train` | [config](/docs/funsd_training/PaddleOCR_training.md) |
| mmOCR | `DBNet` | `CRNN` | [config](/docs/funsd_training/mmocr_training.md) |
| easyOCR | `CRAFT_clr_amp_29500` | `TPS-ResNet-BiLSTM-Attn` | [config](/docs/funsd_training/easyOCR_training.md) |
| tesseract | . | . | . |

## training IAM dataset
| | Detection model | Recognition model | train-config |
| --- | --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` | [config](/docs/iam_training/docTR_training.md) |
| paddleOCR | `MobileNetV3_large_x0_5_pretrained` | `en_PP-OCRv3_rec_train` | [config](/docs/iam_training/PaddleOCR_training.md) |
| mmOCR | `DBNet` | `CRNN` | [config](/docs/iam_training/mmocr_training.md) |
| easyOCR | `CRAFT_clr_amp_29500` | `TPS-ResNet-BiLSTM-Attn` | [config](/docs/iam_training/easyOCR_training.md) |
| tesseract | . | . | . |

## training SROIE dataset
| | Detection model | Recognition model | train-config |
| --- | --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` | [config](/docs/sroie_training/docTR_training.md) |
| paddleOCR | `ResNet50_vd_ssld_pretrained` | `en_PP-OCRv3_rec_train` | [config](/docs/sroie_training/PaddleOCR_training.md) |
| mmOCR | `DBNet` | `ASTER` | [config](/docs/sroie_training/mmocr_training.md) |
| easyOCR | `CRAFT_clr_amp_29500` | . | [config](/docs/sroie_training/easyOCR_training.md) |
| tesseract | . | . | . |