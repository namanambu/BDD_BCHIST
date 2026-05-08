from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    data_root: Path
    result_dir: Path
    img_size: int = 256
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 2
    first_layer_num_kernel: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lambda_subtype: float = 0.5
    patience: int = 10
    device: str = "cuda:0"

    @property
    def metadata_csv(self) -> Path:
        return self.data_root / "BreaKHis_metadata.csv"

    @property
    def split_csv(self) -> Path:
        return self.data_root / "BreaKHis_metadata_split.csv"

    def fold_csv(self, fold: int) -> Path:
        return self.data_root / f"BreaKHis_metadata_fold{fold}.csv"

    def model_path(self, fold: int, tag: str = "rescnn_cascaded") -> Path:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        return self.result_dir / f"{tag}_best_fold{fold}.pth"

    def val_result_path(self, fold: int, tag: str = "rescnn_cascaded") -> Path:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        return self.result_dir / f"val_result_fold{fold}_{tag}.csv"
