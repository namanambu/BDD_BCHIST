from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

'''
    Function: _as_batched_image
    Purpose: Converts a tensor (image) into batched format (number of images in front) and moves it to the selected device.
    Inputs:
      img    - torch.Tensor; image tensor with shape [3, H, W] or [1, 3, H, W].
      device - torch.device; device where the tensor should be stored.
                This has a default in model.py.
    Returns:
      torch.Tensor; batched image tensor with shape [1, 3, H, W] on the selected device.
    Behaviour:
      Adds a batch dimension if the image is unbatched, checks that the final tensor
      has four dimensions, raises an error for invalid shapes, and moves the tensor
      to the selected device.
    '''
def _as_batched_image(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"_as_batched_image only works with shapes [3,H,W] or [1,3,H,W]. Tried {tuple(img.shape)}")
    return img.to(device)

'''
    Function: _as_batched_magn
    Purpose: Converts magnification token into batched format.
    Inputs:
      magn   - torch.Tensor | int; magnification token (see data.py for key)
      device - torch.device; device where the tensor should be stored.
    Returns:
      torch.Tensor; batched tensor with the magnification token.
    Behaviour:
      Converts integer inputs into tensors, adds a batch dimension for scalar
      tokens, moves the tensor to the selected device, and casts it to long type.
    '''
def _as_batched_magn(magn: torch.Tensor | int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(magn):
        magn = torch.tensor(magn, dtype=torch.long)
    if magn.dim() == 0:
        magn = magn.unsqueeze(0)
    return magn.to(device).long()


'''
    Function: generate_gradcam
    Purpose: Generates a Grad-CAM heatmap for one image and its model output.
    Inputs:
      model             - torch.nn.Module; model that returns head1 and head2 logits.
      img               - torch.Tensor; image tensor with shape [3, H, W] or
                          [1, 3, H, W].
      magn              - torch.Tensor | int; magnification token (see data.py for key)
      target_head       - str; output head, either "head1" or "head2".
                            Defaults to head1
      target_class      - int | None; class index. 
                            Defaults to the predicted class.
      target_layer_name - str; name of the convolutional model layer used for
                          Grad-CAM hook registration.
                            Defaults to Conv5
    Returns:
      tuple[np.ndarray, int]; normalized Grad-CAM heatmap with shape [H, W] and
      values [0, 1], the integer is the explained class.
    Behaviour:
      Sets the model to evaluation mode, batches the image and magnification token,
      registers hooks on the selected convolutional layer, runs a forward and backward
      pass for the chosen class score, computes gradient-weighted activations, applies
      ReLU, normalizes and upsamples the heatmap to the input image size, removes the
      hooks, and returns the final heatmap with the explained class.
    '''
def generate_gradcam(
    model,
    img: torch.Tensor,
    magn: torch.Tensor | int,
    target_head: str = "head1",
    target_class: int | None = None,
    target_layer_name: str = "Conv5",
):
    if target_head not in {"head1", "head2"}:
        raise ValueError("target_head has t be 'head1' or 'head2'")

    model.eval()
    device = next(model.parameters()).device
    img_b = _as_batched_image(img, device)
    magn_b = _as_batched_magn(magn, device)

    activations = {}
    gradients = {}

    target_layer = getattr(model, target_layer_name)

    def save_activation(module, inputs, output):
        activations["value"] = output

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.zero_grad(set_to_none=True)
        out1, out2 = model(img_b, magn_b)
        logits = out1 if target_head == "head1" else out2

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        A = activations["value"][0]       # [C,h,w]
        dA = gradients["value"][0]        # [C,h,w]
        weights = dA.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * A).sum(dim=0))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        H, W = img_b.shape[-2:]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return cam.detach().cpu().numpy().astype(np.float32), int(target_class)
    finally:
        forward_handle.remove()
        backward_handle.remove()

'''
    Function: _image_to_numpy
    Purpose: Converts an image from tensor, PIL, or NumPy format into displayable NumPy array.
    Inputs:
      img - torch.Tensor | PIL.Image.Image | np.ndarray; image to convert for visualization.
    Returns:
      np.ndarray; image array clipped to the [0, 1] range.
    Behaviour:
      Converts batched or unbatched PyTorch tensors to channel-last
      format, converts PIL images into scaled NumPy arrays, leaves NumPy arrays alone,
      raises an error for unsupported image types, and clips the result
      to valid display values.
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
        raise TypeError("please use Tensor, PIL.Image, or np.ndarray image types.")
    return np.clip(img_np, 0, 1)


'''
    Function: show_gradcam_on_image
    Purpose: Displays a GradCAM heatmap overlaid on the original image.
    Inputs:
      img   - torch.Tensor | PIL.Image.Image | np.ndarray; image used as the background.
      cam   - torch.Tensor | np.ndarray; GradCAM heatmap to overlay on the image.
      title - str; title displayed above the visualization.
                Defaults to just "GradCAM".
      cmap  - str; Matplotlib colormap used for the heatmap.
                Defaults to "jet" (the blue to red rainbow one).
      alpha - float; transparency level of the heatmap overlay.
                Defaults to 40%.  This is pretty arbitrary.
    Returns:
      N/A
    Behaviour:
      Converts the input image to displayable NumPy format, plots the image
      and GradCAM overlay, adds a colorbar labeled GradCAM Activation,
      and displays the figure.
    '''
def show_gradcam_on_image(img, cam, title="GradCAM", cmap="jet", alpha=0.4):
    img_np = _image_to_numpy(img)
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    heatmap = plt.imshow(cam, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.title(title)
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM Activation", rotation=270, labelpad=15)
    plt.show()
