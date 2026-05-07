from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image


MAGNIFICATION_TO_TOKEN = {40: 0, 100: 1, 200: 2, 400: 3}

'''
Function: magnification_to_token
Purpose: Converts a real magnification value into the corresponding model token.
Inputs:
  magnification - int; the image magnification level (40, 100, 200, or 400).
Returns:
  torch.Tensor; tensor containing the integer token associated with the magnification.
Behaviour:
  Casts the magnification value to an integer, looks up its token using the
  MAGNIFICATION_TO_TOKEN dictionary, and returns the result as a tensor.
  this WILL fail if you pass a magnification value not defined in the dict.
'''
def magnification_to_token(magnification: int) -> torch.Tensor:
    return torch.tensor(MAGNIFICATION_TO_TOKEN[int(magnification)], dtype=torch.long)

'''
Function: image_to_tensor
Purpose: Converts a PIL image into the tensor format model requires.
  Can be skipped if image is already a tensor. 
Inputs:
  img  - PIL.Image.Image; the input image to preprocess.
  size - int; the desired square image size.  (Optional) Default value of 256.
Returns:
  torch.Tensor; the resized RGB image with pixel values in [0, 1].
Behaviour:
  Converts the image to RGB, resizes it to N x N, scales pixel values
  to the [0, 1] range, rearranges the array from HWC to CHW format, and
  converts the result into a PyTorch tensor.
'''
def image_to_tensor(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    return torch.from_numpy(img_np)

'''
Function: load_breakhis_sample
Purpose: Loads one image sample and prepares its inputs.
Inputs:
  metadata_csv - str; path to the CSV file containing image metadata and filepaths.
  index        - int; row index of the sample to load from the metadata file.
  image_size   - int; desired square image size, (Optional) default value of 256.
Returns:
  dict; contains the sample index, metadata row, image path, PIL image,
  image tensor, and magnification token.
Behaviour:
  Reads the metadata CSV, selects the requested row, loads and resizes the
  corresponding image, converts the image into a tensor, converts the
  magnification value into a token, and packages all sample information into
  a dictionary for later use.
'''
def load_breakhis_sample(metadata_csv: str, index: int, image_size: int = 256):
    df = pd.read_csv(metadata_csv)
    row = df.iloc[index]
    img_path = row["filepath"]
    pil_img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    img_tensor = image_to_tensor(pil_img, size=image_size)
    magn_token = magnification_to_token(row["magnification"])

    return {
        "index": index,
        "row": row,
        "image_path": img_path,
        "pil_image": pil_img,
        "image_tensor": img_tensor,
        "magn_token": magn_token,
    }
