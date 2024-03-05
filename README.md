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
| paddleOCR | `en_PP-OCRv3_det_infer` | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | `CRNN` | . |
| easyOCR | . | . | . |
| tesseract | . | . | . |

## training IAM dataset
| | Detection model | Recognition model | train-config |
| --- | --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` | . |
| paddleOCR | `en_PP-OCRv3_det_infer` | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | `CRNN` | . |
| easyOCR | . | . | . |
| tesseract | . | . | . |

## training SROIE dataset
| | Detection model | Recognition model | train-config |
| --- | --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` | . |
| paddleOCR | `en_PP-OCRv3_det_infer` | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | `ASTER` | . |
| easyOCR | . | . | . |
| tesseract | . | . | . |