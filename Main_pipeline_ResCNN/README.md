# BreaKHis ResCNN Encoder+MLP / Cascaded Pipeline

This refactors `Breast_cancer_030426_martin.ipynb` into multiple function files and one example script.

## What was retained

- **Setup**: imports, seed/device helpers, path configuration.
- **Preprocessing**: BreaKHis metadata construction, patient-level train/val split, patient-level 5-fold split, subject-level balancing.
- **Data loading**: `Dataset_BreaK` and `get_dataloader_BreaK`.
- **Model**: ResCNN encoder + MLP heads with magnification embedding.
- **Training/inference**: ResCNN cascaded/gated binary + subtype training and cascaded inference.
- **Evaluation**: 5-fold multi-flag evaluation logic, including image-, magnification-, and patient-level aggregation.

## Structure

```text
breakhis_rescnn_project/
  breakhis_rescnn/
    config.py
    setup_utils.py
    preprocessing.py
    dataset.py
    models/rescnn.py
    train.py
    inference.py
    evaluation_5fold.py
  scripts/example_run_5fold.py
  requirements.txt
```

## How to run

Edit `DATA_ROOT` and `RESULT_DIR` in `scripts/example_run_5fold.py`, then run:

```bash
pip install -r requirements.txt
python scripts/example_run_5fold.py
```

The example script will create metadata, create patient-level 5-fold CSVs, train one model per fold, save validation predictions, and run the retained 5-fold evaluation.
