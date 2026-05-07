from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

'''
    Function: forward_head1
    Purpose: Runs the model and extracts the b/m classification logits.
    Inputs:
      model - torch.nn.Module; model that returns either Head 1 logits or a tuple/list
              containing Head 1 and Head 2 outputs, depending on how it was run.
      image - torch.Tensor; batch of image tensors with shape [B, C, H, W].
      magn  - torch.Tensor; batch of magnification tokens with shape [B].
    Returns:
      torch.Tensor; binary b/m logits with shape [B, 2].
    Behaviour:
      Passes the image and magnification tensors through the model, extracts the
      Head 1 output, checks the result has the expected binary-logit shape, and
      returns the logits.
    '''
def forward_head1(model, image: torch.Tensor, magn: torch.Tensor) -> torch.Tensor:
    out = model(image, magn)
    logits_bin = out[0] if isinstance(out, (tuple, list)) else out
    if logits_bin.dim() != 2 or logits_bin.size(1) != 2:
        raise ValueError(f"forward_head1 should get [B,2], tried {tuple(logits_bin.shape)}")
    return logits_bin

'''
    Function: forward_target_prob
    Purpose: Computes the probability for the target class.
    Inputs:
      model        - torch.nn.Module; model used to generate binary logits (Head1).
      image        - torch.Tensor; batch of image tensors with shape [B, C, H, W].
      magn         - torch.Tensor; batch of magnification tokens with shape [B].
      target_class - int; binary class index to extract (0 -> B, 1 -> M).
    Returns:
      torch.Tensor; probability of the selected target class for each image in the batch,
      with shape [B].
    Behaviour:
      Gets the Head 1 logits, uses softmax to convert them into probabilitie.
      Returns the probability corresponding to the selected target class.
    '''
def forward_target_prob(model, image: torch.Tensor, magn: torch.Tensor, target_class: int) -> torch.Tensor:
    logits = forward_head1(model, image, magn)
    probs = torch.softmax(logits, dim=1)
    return probs[:, target_class]

'''
    Function: make_baseline
    Purpose: Creates the replacement image values used during occlusion (the 
            altered patches we use to occlude bits of the true image).
    Inputs:
      img_batch - torch.Tensor; batch of image tensors with shape [B, C, H, W].
      mode      - str; baseline method to use, either "zero" or "channel_mean".
                Defaults to zero (black).
    Returns:
      torch.Tensor; baseline tensor with the same shape as img_batch.
    Behaviour:
      Set to "channel_mean" computes the mean value of each channel for each image
      and expands it across the full image size. Tells you if you pass an unknown mode
      (Generally zero has been less saturated).
    '''
def make_baseline(img_batch: torch.Tensor, mode: str = "zero") -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(img_batch)
    if mode == "channel_mean":
        ch_mean = img_batch.mean(dim=(2, 3), keepdim=True)
        return ch_mean.expand_as(img_batch)
    raise ValueError(f"I only coded for mode = zero and mean. : ) Feel free to code: {mode}")

'''
    Function: occlusion_grid_delta_p
    Purpose: Computes an occlusion-based attribution map for one Head 1 target class.
    Inputs:
      model         - torch.nn.Module; model used to compute binary class probabilities.
      img           - torch.Tensor; image tensor with shape [C, H, W] or [1, C, H, W].
      magn          - torch.Tensor | int; magnification token for the image.
      target_class  - int; binary class to explain. 0's B, 1's M.  This is probs in the README.
      window_hw     - int; height and width of each square occlusion window (default 32).
      stride_hw     - int; stride between neighboring occlusion windows (default 16).
      baseline_mode - str; method used to fill occluded regions (see make_baseline).
      chunk         - int; number of occluded images evaluated at once (default 64).
    Returns:
      tuple[np.ndarray, np.ndarray, float]; occlusion grid values, interpolated heatmap
      with shape [H, W], and the original target-class probability.
    Behaviour:
      Converts the image and magnification token into batched tensors, computes the
      original target-class probability, slides a square occlusion window across the
      image, replaces each window with baseline values, computes the probability drop
      for each occluded image, stores the delta-p values,
      (Δp = p_original(target_class) - p_occluded(target_class)). interpolates
      the grid to the original image size, and returns both attribution maps and the
      original probability.
    '''
def occlusion_grid_delta_p(
    model,
    img: torch.Tensor,
    magn: torch.Tensor | int,
    target_class: int,
    window_hw: int = 32,
    stride_hw: int = 16,
    baseline_mode: str = "zero",
    chunk: int = 64,
):
    if target_class not in [0, 1]:
        raise ValueError(f"target_class needs to be 0 or 1, not {target_class}")

    model.eval()
    device = next(model.parameters()).device

    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.to(device)

    if not torch.is_tensor(magn):
        magn = torch.tensor(magn, dtype=torch.long)
    if magn.dim() == 0:
        magn = magn.unsqueeze(0)
    magn = magn.to(device).long()

    B, C, H, W = img.shape
    if B != 1:
        raise ValueError("you passed more than 1 image at a time. Check your tensor shape. (B=1)")

    baseline = make_baseline(img, mode=baseline_mode)

    with torch.no_grad():
        p_orig = float(forward_target_prob(model, img, magn, target_class)[0].item())

    nH = (H - window_hw) // stride_hw + 1
    nW = (W - window_hw) // stride_hw + 1
    if nH <= 0 or nW <= 0:
        raise ValueError(f"Window too large. Got nH={nH}, nW={nW} for H={H}, W={W}.")

    grid_vals = np.zeros((nH, nW), dtype=np.float32)
    coords = []
    occluded_batch = []

    def flush():
        nonlocal coords, occluded_batch, grid_vals
        if not occluded_batch:
            return
        x = torch.cat(occluded_batch, dim=0)
        m = magn.expand(x.shape[0])
        with torch.no_grad():
            p_occ = forward_target_prob(model, x, m, target_class).detach().cpu().numpy()
        for (i, j), p in zip(coords, p_occ):
            grid_vals[i, j] = p_orig - float(p)
        coords = []
        occluded_batch = []

    for i in range(nH):
        top = i * stride_hw
        for j in range(nW):
            left = j * stride_hw
            x_occ = img.clone()
            x_occ[:, :, top:top + window_hw, left:left + window_hw] = baseline[:, :, top:top + window_hw, left:left + window_hw]
            occluded_batch.append(x_occ)
            coords.append((i, j))
            if len(occluded_batch) >= chunk:
                flush()
    flush()

    ys = np.array([i * stride_hw + (window_hw / 2.0) for i in range(nH)], dtype=np.float32)
    xs = np.array([j * stride_hw + (window_hw / 2.0) for j in range(nW)], dtype=np.float32)
    interp = RegularGridInterpolator((ys, xs), grid_vals, method="linear", bounds_error=False, fill_value=0.0)

    Y, X = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    pts = np.stack([Y.ravel(), X.ravel()], axis=1)
    heat = interp(pts).reshape(H, W).astype(np.float32)
    return grid_vals, heat, p_orig

'''
    Function: run_occlusion_head1
    Purpose: Runs class-conditional occlusion analysis for Head 1.
    Inputs:
      model         - torch.nn.Module; model used to compute Head 1 predictions.
      img           - torch.Tensor; image tensor with shape [C, H, W] or [1, C, H, W].
      magn          - torch.Tensor | int; magnification token for the image.
      target_class  - int | None; binary class to explain. Defaults to predicted
                        class.
      window_hw     - int; height and width of each square occlusion window.
                        Defaults to 32.
      stride_hw     - int; stride between neighboring occlusion windows.
                        Defaults to 16.
      baseline_mode - str; method used to fill occluded regions.
                        Defaults tsezo.
      chunk         - int; number of occluded images evaluated at once.
                        Defaults to 64.
    Returns:
      dict; contains the predicted class, explained target class, target probability,
      original probability, occlusion heatmap, coarse occlusion grid, and display image.
    Behaviour:
      Prepares the image and magnification token, computes the Head 1 prediction and
      probability for the predicted class, runs the occlusion delta-p calculation,
      converts the image into NumPy format for displaying it, and returns all prediction
      and attribution outputs in a dictionary. QUICKNOTE: You can strap it to a particular
      class (B/M) instead of it being the predicted class.  That way the heatmap is always
      "evidence suggesting malignancy" for example, which I think makes sense. But that
      was deemed confusing during presentation and thus made default = None.
    '''
def run_occlusion_head1(
    model,
    img: torch.Tensor,
    magn: torch.Tensor | int,
    target_class: int | None = None,
    window_hw: int = 32,
    stride_hw: int = 16,
    baseline_mode: str = "zero",
    chunk: int = 64,
):
    model.eval()
    device = next(model.parameters()).device

    if img.dim() == 3:
        img_b = img.unsqueeze(0)
    else:
        img_b = img
    img_b = img_b.to(device)

    if not torch.is_tensor(magn):
        magn = torch.tensor(magn, dtype=torch.long)
    if magn.dim() == 0:
        magn_b = magn.unsqueeze(0)
    else:
        magn_b = magn
    magn_b = magn_b.to(device).long()

    with torch.no_grad():
        logits = forward_head1(model, img_b, magn_b)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        if target_class is None:
            target_class = pred
        p_target = float(probs[0, target_class].item())

    grid, heat, p_orig = occlusion_grid_delta_p(
        model=model,
        img=img_b,
        magn=magn_b,
        target_class=target_class,
        window_hw=window_hw,
        stride_hw=stride_hw,
        baseline_mode=baseline_mode,
        chunk=chunk,
    )

    img_np = img_b.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    return {
        "pred": pred,
        "target_class": int(target_class),
        "p_target": p_target,
        "p_orig": p_orig,
        "heat": heat,
        "grid": grid,
        "img_np": img_np,
    }

'''
    Function: _image_to_numpy
    Purpose: Converts an image into a displayable NumPy array.
    Inputs:
      img - torch.Tensor | PIL.Image.Image | np.ndarray; image to convert for visualization.
    Returns:
      np.ndarray; image array clipped to the [0, 1] range.
    Behaviour:
      Converts from channel-first to channel-last format to produce NumPy arrays 
      for displayable values.  You'll get an error if its not one of the three types
      I've hardcoded.  But you can add another conditional to handle a new type if 
      you need to. 
    '''
def _image_to_numpy(img) -> np.ndarray:
    if torch.is_tensor(img):
        if img.dim() == 4:
            img = img.squeeze(0)
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    elif isinstance(img, Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        raise TypeError("_image_to_numpy works for tensor, PIL.Image, or np.ndarray.")
    return np.clip(img_np, 0, 1)

'''
    Function: show_occlusion_on_image
    Purpose: Displays an occlusion heatmap overlaid on the original image.
    Inputs:
      img   - torch.Tensor | PIL.Image.Image | np.ndarray; image used as the background.
      heat  - torch.Tensor | np.ndarray; occlusion attribution heatmap to overlay.
      title - str; title displayed above the visualization. Defaults to just "Occlusion"
      cmap  - str; Matplotlib colormap used for the heatmap. Defaults to jet (blue to red rainbow)
      alpha - float; transparency level of the heatmap overlay (arbitrarily defaults to 45%)
    Returns:
      None.
    Behaviour:
      Converts the heatmap and image into NumPy display format, plots the image with
      the occlusion heatmap overlay, scales the color range to the heatmap minimum
      and maximum, removes axis markings, adds a labeled colorbar, and displays.
      Do note that I was lazy and use this function for the difference visualization as well...
      In those scenarios you may want to actually specify the title so you don't get confused.
    '''
def show_occlusion_on_image(img, heat, title="Occlusion", cmap="jet", alpha=0.45):
    if torch.is_tensor(heat):
        heat = heat.detach().cpu().numpy()
    img_np = _image_to_numpy(img)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    overlay = plt.imshow(heat, cmap=cmap, alpha=alpha, vmin=float(np.min(heat)) - 1e-8, vmax=float(np.max(heat)) + 1e-8)
    plt.axis("off")
    plt.title(title)
    cbar = plt.colorbar(overlay, fraction=0.046, pad=0.04)
    cbar.set_label("Occlusion Attribution", rotation=270, labelpad=15)
    plt.show()
