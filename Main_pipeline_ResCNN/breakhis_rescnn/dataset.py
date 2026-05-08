from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler


tumor_map = {"b": 0, "m": 1}
subtype_map = {"a": 0, "f": 1, "pt": 2, "ta": 3, "dc": 4, "lc": 5, "mc": 6, "pc": 7}
magn_map = {40: 0, 100: 1, 200: 2, 400: 3}
idx_to_magn = {v: k for k, v in magn_map.items()}


class Dataset_BreaK(Dataset):
    def __init__(self, df, augment: bool = True, img_size: int = 256):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.img_size = img_size
        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15, fill=(255, 255, 255)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.item()
        row = self.df.iloc[index]

        img = Image.open(Path(row.filepath)).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        if self.augment:
            img = self.augment_transform(img)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = torch.from_numpy(img_np)

        ttype = tumor_map[str(row.tumor_type).lower()]
        stype = subtype_map[str(row.subtype).lower()]
        magn = magn_map[int(row.magnification)]

        return {
            "image": img_tensor,
            "tumor_type": torch.tensor(ttype, dtype=torch.long),
            "subtype": torch.tensor(stype, dtype=torch.long),
            "magn": torch.tensor(magn, dtype=torch.long),
        }


def get_dataloader_BreaK(
    df,
    mode: str = "train",
    batch_size: int = 4,
    num_workers: int | None = None,
    shuffle: bool = True,
    sample_ratio: float = 1.0,
    data_aug: bool = True,
    img_size: int = 256,
):
    dataset = Dataset_BreaK(df, augment=data_aug, img_size=img_size)
    if num_workers is None:
        num_workers = min(2, batch_size)

    num_samples = int(len(dataset) * sample_ratio)
    indices = torch.arange(len(dataset))[:num_samples]

    if mode.lower() == "test":
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    sampler = SubsetRandomSampler(indices) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
    )
