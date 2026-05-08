# Swin Transform BreaKHis Classification

Swin Transformer (swin-tiny) was explored as an alternative architecture for binary benign/malignant
classification histopathology dataset. 

## Structure

    swin/
        swinFunctions.py   # includes initialization, data parsing, data loading, model initialization, model training, and model evaluation
        swinBreakHis.ipynb  # full training run with logged outputs and figures
        requirements.txt
        README.md

## Setup (Google Colab)

```python
# 1. Import your custom library
import swinFunctions as sf
import pandas as pd
import torch

# 2. Initialize your configuration
config = sf.get_training_config()

# 3. Load Metadata and Create DataLoaders
df = pd.read_csv(config['METADATA_PATH'])
train_transform, val_transform = sf.get_transforms()

train_loader, val_loader = sf.create_dataloaders(
    df, 
    train_transform, 
    val_transform, 
    batch_size=config['BATCH_SIZE']
)
```

## Quickstart

```python
# Load the model architecture and the best saved weights
model = sf.load_swin_transformer(config['NUM_CLASSES'], config['DEVICE'])
model.load_state_dict(torch.load('best_swin_model.pth'))

# 1. Image-Level Evaluation (CM and ROC Plots)
conf_matrix, auc_score, fpr, tpr = sf.evaluate_model(
    model, 
    val_loader, 
    "Validation Set", 
    config['DEVICE']
)

# 2. Patient-Level Evaluation (Majority Voting)
patient_results = sf.perform_patient_level_analysis(
    model, 
    val_loader, 
    df, 
    config['DEVICE']
)
```

## Example

See `swinBreakHis.ipynb` for the full training run with logged outputs
and figures, including epoch-by-epoch metrics, confusion matrices, and
training curves.

## Key Results

| Metric | Value |
|--------|-------|
| Image-level AUC | 0.83 |
| False negative rate | 2.1% |
| Epochs to convergence | 5 |
| Patient-level accuracy | 94.1% (17 subjects) |
| Malignant Patients Identified | 11/11 |

## Notes

- Uses a hierarchical windowed self-attention mechanism, which is better at preserving local spatial hierarchies in histopathology slides compared to standard ViT.
- The checkpoint `best_swin_model.pth` corresponds to epoch 5 (best val AUC).
- The model displays high sensitivity (perfect at the patient level), making it an effective conservative screening tool that ensures no malignant cases are missed.
- Final diagnosis is determined via Majority Voting; if >50% of images from a single subject are flagged, the patient is classified as Malignant.
- All core logic resides in swinFunctions.py, allowing the Jupyter Notebook to remain focused on visualization and experimentation.
- Paths need to be altered to match your file management for your locally downloaded dataset

For questions contact: namanam1@jh.edu
