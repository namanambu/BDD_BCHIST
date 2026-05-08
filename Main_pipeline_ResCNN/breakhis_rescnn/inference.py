from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import get_dataloader_BreaK, idx_to_magn
from .models import ResCNNEncoderMLP
from .setup_utils import get_device


def predict_rescnn_cascaded(
    metadata_csv: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    partition: str = "val",
    batch_size: int = 1,
    img_size: int = 256,
    num_workers: int = 1,
    first_layer_num_kernel: int = 64,
    device: str | None = None,
    mask_token: bool = False,
):
    device = get_device(device)
    model = ResCNNEncoderMLP(img_ch=3, first_layer_numKernel=first_layer_num_kernel).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = pd.read_csv(metadata_csv)
    val_df = data[data["partition"] == partition].copy().reset_index(drop=True)
    loader = get_dataloader_BreaK(val_df, mode="test", batch_size=batch_size, num_workers=num_workers, shuffle=False, data_aug=False, img_size=img_size)

    rows = []
    offset = 0
    with torch.inference_mode():
        for sample in tqdm(loader, desc="Inference"):
            img = sample["image"].to(device, non_blocking=True)
            magn = sample["magn"].to(device, non_blocking=True)
            y_bm = sample["tumor_type"].long().to(device, non_blocking=True)
            y_sub = sample["subtype"].long().to(device, non_blocking=True)

            out1, out2 = model(img, magn, mask_token=mask_token)
            p1 = F.softmax(out1, dim=1)
            p2 = F.softmax(out2, dim=1)

            for b in range(img.shape[0]):
                pred_bm = int(p1[b].argmax().item())
                if pred_bm == 0:
                    pred_sub = int(p2[b, :4].argmax().item())
                else:
                    pred_sub = int(p2[b, 4:].argmax().item() + 4)

                meta = val_df.iloc[offset + b].to_dict()
                row = {
                    **meta,
                    "gt_bm": int(y_bm[b].item()),
                    "gt_sub": int(y_sub[b].item()),
                    "pred_bm": pred_bm,
                    "pred_sub_cascaded": pred_sub,
                    "pred_prob_bm_0": float(p1[b, 0].cpu()),
                    "pred_prob_bm_1": float(p1[b, 1].cpu()),
                    "magnification_from_token": idx_to_magn[int(magn[b].cpu())],
                }
                for c in range(8):
                    row[f"pred_prob_sub_{c}"] = float(p2[b, c].cpu())
                rows.append(row)
            offset += img.shape[0]

    result_df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    return result_df


predict_rescnn_encoder_mlp = predict_rescnn_cascaded
