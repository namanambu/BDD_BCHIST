import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def build_breakhis_metadata(data_root: str | Path, save_csv: str | Path | None = None) -> pd.DataFrame:
    """Traverse BreaKHis folders and create image-level metadata."""
    data_root = Path(data_root)
    records = []

    for tumor_type in ["benign", "malignant"]:
        tumor_dir = data_root / tumor_type
        if not tumor_dir.exists():
            continue

        sob_dir = tumor_dir / "SOB"
        if not sob_dir.exists():
            continue

        for subtype in os.listdir(sob_dir):
            subtype_dir = sob_dir / subtype
            if not subtype_dir.is_dir():
                continue

            for subject_folder in os.listdir(subtype_dir):
                subject_dir = subtype_dir / subject_folder
                if not subject_dir.is_dir():
                    continue

                m = re.search(r"(?P<year>\d{2})[-_](?P<slide>\d+[A-Z]{0,3})$", subject_folder)
                subject_id = f"{m.group('year')}-{m.group('slide')}" if m else subject_folder

                for mag_folder in os.listdir(subject_dir):
                    mag_dir = subject_dir / mag_folder
                    if not mag_dir.is_dir():
                        continue

                    try:
                        magnification = int("".join(filter(str.isdigit, mag_folder)))
                    except Exception:
                        magnification = None

                    for fname in os.listdir(mag_dir):
                        if not fname.lower().endswith(".png"):
                            continue

                        fpath = mag_dir / fname
                        name_parts = fname.split("_")
                        tumor_flag = name_parts[1].lower() if len(name_parts) > 1 else tumor_type[0]
                        subtype_abbr = name_parts[2].split("-")[0].lower() if len(name_parts) > 2 else subtype.lower()

                        mag_in_name = None
                        if "-" in fname:
                            for s in fname.split("-"):
                                if s.isdigit() and s in ["40", "100", "200", "400"]:
                                    mag_in_name = int(s)
                                    break
                        magnification_final = mag_in_name or magnification

                        records.append({
                            "filepath": str(fpath),
                            "tumor_type": tumor_flag,
                            "subtype": subtype_abbr,
                            "magnification": magnification_final,
                            "subject_id": subject_id,
                        })

    df = pd.DataFrame(records)
    if save_csv is None:
        save_csv = data_root / "BreaKHis_metadata.csv"
    df.to_csv(save_csv, index=False)
    return df


def make_patient_train_val_split(
    metadata_csv: str | Path,
    output_csv: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Create one patient-level train/val split to avoid image-level leakage."""
    random.seed(seed)
    data = pd.read_csv(metadata_csv)
    unique_subjects = data["subject_id"].unique().tolist()
    random.shuffle(unique_subjects)

    n_train = int(train_ratio * len(unique_subjects))
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    def assign_partition(row):
        sid = row["subject_id"]
        if sid in train_subjects:
            return "train"
        if sid in val_subjects:
            return "val"
        raise ValueError(f"Subject {sid} was not assigned to a split.")

    data["partition"] = data.apply(assign_partition, axis=1)
    data.to_csv(output_csv, index=False)
    return data


def make_patient_5fold_splits(
    metadata_csv: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Path]:
    """Create patient-level 5-fold CSVs named BreaKHis_metadata_fold{fold}.csv."""
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(metadata_csv)
    subjects = data["subject_id"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    output_paths = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(subjects), start=1):
        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        def assign_partition(row):
            sid = row["subject_id"]
            if sid in train_subjects:
                return "train"
            if sid in val_subjects:
                return "val"
            raise ValueError("Subject ID not found in any fold")

        fold_data = data.copy()
        fold_data["partition"] = fold_data.apply(assign_partition, axis=1)
        output_path = output_dir / f"BreaKHis_metadata_fold{fold}.csv"
        fold_data.to_csv(output_path, index=False)
        output_paths.append(output_path)
    return output_paths


def balance_by_subject(
    df: pd.DataFrame,
    partition_col: str = "partition",
    tumor_col: str = "tumor_type",
    subject_col: str = "subject_id",
    mode: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """Balance benign/malignant cases at the subject level within one partition."""
    rng = np.random.default_rng(seed)
    df_part = df[df[partition_col] == mode].copy()
    subj_type = df_part[[subject_col, tumor_col]].drop_duplicates()

    benign_subjects = subj_type[subj_type[tumor_col].str.lower() == "b"][subject_col].tolist()
    malig_subjects = subj_type[subj_type[tumor_col].str.lower() == "m"][subject_col].tolist()
    n_keep = min(len(benign_subjects), len(malig_subjects))

    keep_subjects = set(rng.choice(benign_subjects, n_keep, replace=False).tolist()) | set(
        rng.choice(malig_subjects, n_keep, replace=False).tolist()
    )
    return df_part[df_part[subject_col].isin(keep_subjects)].reset_index(drop=True)
