# ocr-benchmark-result
ocr-benchmark-result
# models
- before training/finetuning (for every dataset)

| | Detection model | Recognition model |
| --- | --- | --- |
| docTR | `db_resnet50` | `crnn_vgg16_bn` |
| paddleOCR | `en_PP-OCRv3_det_infer` | `en_PP-OCRv4_rec_infer` |
| mmOCR | `DBNet` | `CRNN` |
| easyOCR | . | . |
| tesseract | . | . |

### training FUNSD dataset
| | Detection model | det-config | Recognition model | rec-config |
| --- | --- | --- | --- | --- |
| docTR | `db_resnet50` | . | `crnn_vgg16_bn` | . |
| paddleOCR | `en_PP-OCRv3_det_infer` | . | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | . | `CRNN` | . |
| easyOCR | . | . | . | . |
| tesseract | . | . | . | . |

### training IAM dataset
| | Detection model | det-config | Recognition model | rec-config |
| --- | --- | --- | --- | --- |
| docTR | `db_resnet50` | . | `crnn_vgg16_bn` | . |
| paddleOCR | `en_PP-OCRv3_det_infer` | . | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | . | `CRNN` | . |
| easyOCR | . | . | . | . |
| tesseract | . | . | . | . |

### training SROIE dataset
| | Detection model | det-config | Recognition model | rec-config |
| --- | --- | --- | --- | --- |
| docTR | `db_resnet50` | . | `crnn_vgg16_bn` | . |
| paddleOCR | `en_PP-OCRv3_det_infer` | . | `en_PP-OCRv4_rec_infer` | . |
| mmOCR | `DBNet` | . | `CRNN` | . |
| easyOCR | . | . | . | . |
| tesseract | . | . | . | . |