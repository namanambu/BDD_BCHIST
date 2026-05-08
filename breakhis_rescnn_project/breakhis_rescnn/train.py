from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .dataset import get_dataloader_BreaK
from .models import ResCNNEncoderMLP
from .preprocessing import balance_by_subject
from .setup_utils import get_device, set_seed


def gated_subtype_loss(out2, label_bm, label_sub, criterion_sub):
    """Subtype CE computed within benign classes 0-3 and malignant classes 4-7."""
    is_b = label_bm == 0
    is_m = label_bm == 1
    loss_b = out2.new_tensor(0.0)
    loss_m = out2.new_tensor(0.0)
    if is_b.any():
        loss_b = criterion_sub(out2[is_b, :4], label_sub[is_b])
    if is_m.any():
        loss_m = criterion_sub(out2[is_m, 4:], label_sub[is_m] - 4)
    nb = is_b.sum().clamp(min=1)
    nm = is_m.sum().clamp(min=1)
    return (loss_b * nb + loss_m * nm) / (nb + nm)


def train_rescnn_cascaded(
    metadata_csv: str | Path,
    model_path: str | Path,
    fold: int | None = None,
    batch_size: int = 8,
    img_size: int = 256,
    num_workers: int = 2,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_subtype: float = 0.5,
    patience: int = 10,
    first_layer_num_kernel: int = 64,
    device: str | None = None,
    seed: int = 42,
    balance_train: bool = True,
):
    set_seed(seed)
    device = get_device(device)
    data = pd.read_csv(metadata_csv)

    train_df = data[data["partition"] == "train"].copy()
    val_df = data[data["partition"] == "val"].copy()
    if balance_train:
        train_df = balance_by_subject(data, mode="train", seed=seed)

    train_loader = get_dataloader_BreaK(train_df, mode="train", batch_size=batch_size, num_workers=num_workers, data_aug=True, img_size=img_size)
    val_loader = get_dataloader_BreaK(val_df, mode="test", batch_size=batch_size, num_workers=num_workers, data_aug=False, img_size=img_size)

    model = ResCNNEncoderMLP(img_ch=3, first_layer_numKernel=first_layer_num_kernel).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sub = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-6)

    best_val_loss = float("inf")
    patience_left = patience
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(epochs), desc=f"Training fold {fold}" if fold else "Training"):
        model.train()
        training_loss = 0.0

        for sample in train_loader:
            img = sample["image"].to(device, non_blocking=True)
            magn = sample["magn"].to(device, non_blocking=True)
            label_bm = sample["tumor_type"].long().to(device, non_blocking=True)
            label_sub = sample["subtype"].long().to(device, non_blocking=True)

            out1, out2 = model(img, magn)
            loss_sub = gated_subtype_loss(out2, label_bm, label_sub, criterion_sub)
            loss = criterion_cls(out1, label_bm) + lambda_subtype * loss_sub

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for sample in val_loader:
                img = sample["image"].to(device, non_blocking=True)
                magn = sample["magn"].to(device, non_blocking=True)
                label_bm = sample["tumor_type"].long().to(device, non_blocking=True)
                label_sub = sample["subtype"].long().to(device, non_blocking=True)

                out1, out2 = model(img, magn)
                loss_sub = gated_subtype_loss(out2, label_bm, label_sub, criterion_sub)
                loss = criterion_cls(out1, label_bm) + lambda_subtype * loss_sub
                validation_loss += loss.item()

        val_loss = validation_loss / max(len(val_loader), 1)
        train_loss = training_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch + 1}/{epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        scheduler.step(val_loss)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        patience_left -= 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model updated: {model_path}")
            patience_left = patience

        if patience_left == 4:
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        if patience_left == 0:
            print("Early stopping.")
            break

    return model_path


# Alias for the requested encoder+MLP training block. It uses the same retained ResCNN encoder+MLP model
# and the gated cascaded subtype constraint from the original notebook.
train_rescnn_encoder_mlp = train_rescnn_cascaded
