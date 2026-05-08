"""Example script for BreaKHis ResCNN encoder+MLP / cascaded training.

Edit DATA_ROOT and RESULT_DIR before running.
"""
from pathlib import Path

from breakhis_rescnn.config import ProjectConfig
from breakhis_rescnn.evaluation_5fold import evaluate_5fold, load_5fold_results, print_5fold_summary
from breakhis_rescnn.inference import predict_rescnn_cascaded
from breakhis_rescnn.preprocessing import build_breakhis_metadata, make_patient_5fold_splits, make_patient_train_val_split
from breakhis_rescnn.train import train_rescnn_cascaded


DATA_ROOT = Path("/content/drive/MyDrive/Zhuoyao/Course/BME_data_design/data/BreaKHis_v1/histology_slides/breast")
RESULT_DIR = Path("/content/drive/MyDrive/Zhuoyao/Course/BME_data_design/result/rescnn_cascaded")

cfg = ProjectConfig(data_root=DATA_ROOT, result_dir=RESULT_DIR, batch_size=8, epochs=50, device="cuda:0")

# 1) Setup + preprocessing: create metadata and patient-level split files.
# Skip build_breakhis_metadata if BreaKHis_metadata.csv already exists.
if not cfg.metadata_csv.exists():
    build_breakhis_metadata(cfg.data_root, cfg.metadata_csv)

# Optional single train/val split retained from the original notebook.
make_patient_train_val_split(cfg.metadata_csv, cfg.split_csv, train_ratio=0.8, seed=cfg.seed)

# Required 5-fold patient-level split.
make_patient_5fold_splits(cfg.metadata_csv, cfg.data_root, n_splits=5, seed=cfg.seed)

# 2) Train and run inference for each fold.
for fold in range(1, 6):
    fold_csv = cfg.fold_csv(fold)
    model_path = cfg.model_path(fold, tag="rescnn_cascaded")
    val_csv = cfg.val_result_path(fold, tag="rescnn_cascaded")

    train_rescnn_cascaded(
        metadata_csv=fold_csv,
        model_path=model_path,
        fold=fold,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        num_workers=cfg.num_workers,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lambda_subtype=cfg.lambda_subtype,
        patience=cfg.patience,
        first_layer_num_kernel=cfg.first_layer_num_kernel,
        device=cfg.device,
        seed=cfg.seed,
        balance_train=True,
    )

    predict_rescnn_cascaded(
        metadata_csv=fold_csv,
        model_path=model_path,
        output_csv=val_csv,
        partition="val",
        batch_size=1,
        img_size=cfg.img_size,
        num_workers=1,
        first_layer_num_kernel=cfg.first_layer_num_kernel,
        device=cfg.device,
        mask_token=False,
    )

# 3) 5-fold-only evaluation retained from the original notebook.
val_all_df = load_5fold_results(cfg.result_dir, pattern="val_result_fold{fold}_rescnn_cascaded.csv")
results = evaluate_5fold(val_all_df)
print_5fold_summary(results)
