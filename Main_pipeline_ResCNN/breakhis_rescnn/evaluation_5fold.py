from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


SUB_PROB_COLS = [f"pred_prob_sub_{i}" for i in range(8)]


def load_5fold_results(base_path: str | Path, pattern: str = "val_result_fold{fold}_rescnn_cascaded.csv") -> pd.DataFrame:
    base_path = Path(base_path)
    dfs = []
    for fold in range(1, 6):
        path = base_path / pattern.format(fold=fold)
        df = pd.read_csv(path)
        df["fold"] = fold
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thr[best_idx]
    y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "AUC": auc,
        "Best_thresh": best_thresh,
        "ACC": acc,
        "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        "FNR": fn / (fn + tp) if (fn + tp) else np.nan,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def multiclass_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    per_class_auc = {}
    for c in range(y_prob.shape[1]):
        if np.sum(y_true == c) == 0 or len(np.unique((y_true == c).astype(int))) < 2:
            per_class_auc[c] = np.nan
        else:
            per_class_auc[c] = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "macro_AUC": float(np.mean(valid_aucs)) if valid_aucs else np.nan,
        "ACC": accuracy_score(y_true, y_pred),
        "per_class_auc": per_class_auc,
        "valid_classes": [k for k, v in per_class_auc.items() if not np.isnan(v)],
    }


def evaluate_image_level(val_df: pd.DataFrame) -> dict:
    binary = binary_metrics(val_df["gt_bm"], val_df["pred_prob_bm_1"])
    subtype = multiclass_metrics(val_df["gt_sub"], val_df[SUB_PROB_COLS])
    return {"binary": binary, "subtype": subtype}


def evaluate_magnification_level(val_df: pd.DataFrame) -> dict:
    mag_level = val_df.groupby(["subject_id", "magnification"])["pred_prob_bm_1"].mean().reset_index()
    patient_gt = val_df.groupby("subject_id")["gt_bm"].first().reset_index()
    mag_level = mag_level.merge(patient_gt, on="subject_id")

    binary_rows = []
    for mag in sorted(mag_level["magnification"].unique()):
        df_mag = mag_level[mag_level["magnification"] == mag]
        if len(np.unique(df_mag["gt_bm"])) < 2:
            continue
        row = {"magnification": mag, "n_patients": len(df_mag)}
        row.update(binary_metrics(df_mag["gt_bm"], df_mag["pred_prob_bm_1"]))
        binary_rows.append(row)

    mag_level_sub = val_df.groupby(["subject_id", "magnification"])[SUB_PROB_COLS].mean().reset_index()
    patient_gt_sub = val_df.groupby("subject_id")["gt_sub"].first().reset_index()
    mag_level_sub = mag_level_sub.merge(patient_gt_sub, on="subject_id")

    subtype_rows = []
    for mag in sorted(mag_level_sub["magnification"].unique()):
        df_mag = mag_level_sub[mag_level_sub["magnification"] == mag]
        metrics = multiclass_metrics(df_mag["gt_sub"], df_mag[SUB_PROB_COLS])
        subtype_rows.append({"magnification": mag, "n_patients": len(df_mag), **metrics})

    return {"binary_by_magnification": pd.DataFrame(binary_rows), "subtype_by_magnification": pd.DataFrame(subtype_rows)}


def evaluate_patient_level(val_df: pd.DataFrame) -> dict:
    mag_level = val_df.groupby(["subject_id", "magnification"])["pred_prob_bm_1"].mean().reset_index()
    patient_level = mag_level.groupby("subject_id")["pred_prob_bm_1"].mean().reset_index()
    patient_gt = val_df.groupby("subject_id")["gt_bm"].first().reset_index()
    patient_df = patient_level.merge(patient_gt, on="subject_id")
    binary = binary_metrics(patient_df["gt_bm"], patient_df["pred_prob_bm_1"])
    binary["classification_report"] = classification_report(
        patient_df["gt_bm"].astype(int),
        (patient_df["pred_prob_bm_1"] >= binary["Best_thresh"]).astype(int),
        digits=4,
    )

    mag_level_sub = val_df.groupby(["subject_id", "magnification"])[SUB_PROB_COLS].mean().reset_index()
    patient_level_sub = mag_level_sub.groupby("subject_id")[SUB_PROB_COLS].mean().reset_index()
    patient_gt_sub = val_df.groupby("subject_id")["gt_sub"].first().reset_index()
    patient_df_sub = patient_level_sub.merge(patient_gt_sub, on="subject_id")
    subtype = multiclass_metrics(patient_df_sub["gt_sub"], patient_df_sub[SUB_PROB_COLS])
    return {"binary": binary, "subtype": subtype}


def evaluate_5fold(val_all_df: pd.DataFrame) -> dict:
    return {
        "image_level": evaluate_image_level(val_all_df),
        "magnification_level": evaluate_magnification_level(val_all_df),
        "patient_level": evaluate_patient_level(val_all_df),
    }


def print_5fold_summary(results: dict) -> None:
    print("\n=== 5-fold image-level binary ===")
    print(pd.Series(results["image_level"]["binary"]).round(4))
    print("\n=== 5-fold image-level subtype ===")
    print(results["image_level"]["subtype"])
    print("\n=== 5-fold magnification-level binary ===")
    print(results["magnification_level"]["binary_by_magnification"].round(4))
    print("\n=== 5-fold magnification-level subtype ===")
    print(results["magnification_level"]["subtype_by_magnification"])
    print("\n=== 5-fold patient-level binary ===")
    print(pd.Series({k: v for k, v in results["patient_level"]["binary"].items() if k != "classification_report"}).round(4))
    print("\nClassification report:\n", results["patient_level"]["binary"]["classification_report"])
    print("\n=== 5-fold patient-level subtype ===")
    print(results["patient_level"]["subtype"])
