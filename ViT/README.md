# ViT — Vision Transformer for BreaKHis Classification

Vision Transformer (ViT-Small) implementation for binary benign/malignant
classification and 8-way subtype classification on the BreaKHis breast cancer
histopathology dataset. Includes learnable magnification token fusion matching
the ResNet multimodal baseline architecture.

## Structure

    ViT/
        __init__.py      # package exports
        model.py         # ViT_BreaKHis architecture and load_vit()
        data.py          # BreaKHisDataset, get_transforms(), load_metadata()
        train.py         # MultiTaskLoss, train_one_epoch(), validate()
        evaluate.py      # plot_figures(), plot_confusion_matrix(),
                         # plot_training_curves(), plot_subtype_accuracy()
        ViT_training.ipynb  # full training run with logged outputs and figures
        requirements.txt
        README.md

## Setup (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r /content/drive/MyDrive/BDD_BCHIST/ViT/requirements.txt

import sys
sys.path.append('/content/drive/MyDrive/BDD_BCHIST/ViT')

from vit import (
    load_vit,
    BreaKHisDataset,
    get_transforms,
    MultiTaskLoss,
    validate,
    plot_figures,
)
from vit.data import load_metadata
```

## Quickstart

```python
import torch
from torch.utils.data import DataLoader

METADATA_CSV = '/content/drive/MyDrive/BreaKHis_v1/histology_slides/breast/BreaKHis_metadata.csv'
WEIGHTS_PATH = '/content/drive/MyDrive/vit_breakhis_best.pth'

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model      = load_vit(WEIGHTS_PATH, device=device)
val_df     = load_metadata(METADATA_CSV, partition='val')
val_loader = DataLoader(BreaKHisDataset(val_df, get_transforms('val')),
                        batch_size=32, shuffle=False)

criterion   = MultiTaskLoss()
val_metrics = validate(model, val_loader, criterion, device, epoch=0)
print('Val AUC:', val_metrics['binary_auc'])
```

## Example

See `ViT_training.ipynb` for the full training run with logged outputs
and figures, including epoch-by-epoch metrics, confusion matrices, and
training curves.

## Key Results

| Metric | Value |
|--------|-------|
| Image-level AUC | 0.932 |
| Image-level accuracy | 94.6% |
| Sensitivity | 97.9% |
| False negative rate | 2.1% |
| Specificity | 88.0% |
| Patient-level accuracy | 94.1% (17 subjects) |
| Subtype val accuracy | 42.1--47.7% |

## Notes

- Training used early stopping based on validation AUC, terminating after
  10 epochs upon observing no meaningful improvement beyond epoch 5.
- The checkpoint `vit_breakhis_best.pth` corresponds to epoch 5 (best val AUC).
- Subtype classification underperforms due to class imbalance and ViT's
  data requirements on the relatively small BreaKHis training set.
- This code is NOT compatible with the ResNet baseline or Swin Transformer.
- All paths assume Google Colab with Google Drive mounted. Update paths
  to match your local directory structure if running outside Colab.

For questions contact: ssingha9@jh.edu
