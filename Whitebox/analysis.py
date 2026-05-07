import numpy as np

'''
    Function: normalize_abs_0_100
    Purpose: Normalizes an input array by absolute magnitude between 0 and 100.
    Inputs:
      x   - array-like; input values to normalize.
      eps - float; small value added to the denominator(div by 0 is baddd).
            Defaults to 1.0 x 10^-8 (I would just leave it alone).
    Returns:
      np.ndarray; absolute-value-normalized array with values scaled relative to 100.
    Behaviour:
      Converts input values to a float32 NumPy array and divides them by the
      max element, adds the eps, and multiplies by 100 so the largest absolute
      magnitude is ~100.
    '''
def normalize_abs_0_100(x, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float32)
    return np.abs(x) / (np.max(np.abs(x)) + eps) * 100.0

'''
    Function: saliency_difference
    Purpose: Computes the absolute difference between Grad-CAM and occlusion heatmaps.
    Inputs:
      cam - array-like; Grad-CAM heatmap values.
      occ - array-like; Occlusion heatmap values.
    Returns:
      np.ndarray; absolute difference between the normalized heatmaps at each pixel.
    Behaviour:
      Normalizes both saliency maps to a 0 to 100, subtracts the normalized
      GradCAM map from the normalized occlusion map, takes the absolute val,
      and returns the resulting difference map. In theory would work for other
      methods.
    '''
def saliency_difference(cam, occ):
    cam_norm = normalize_abs_0_100(cam)
    occ_norm = normalize_abs_0_100(occ)
    return np.abs(occ_norm - cam_norm)

'''
    Function: saliency_pearson
    Purpose: Computes Pearson correlation coef between GradCAM and occlusion heatmaps.
    Inputs:
      cam - array-like; GradCAM heatmap values.
      occ - array-like; Occlusion heatmap values.
    Returns:
      float; Pearson correlation coefficient between the normalized heatmaps.
    Behaviour:
      Normalizes both saliency maps between 0 and 100, flattens them into 1D
      arrays, computes their Pearson value using NumPy, and returns the Pearson
      as a float.  Again, in theory not bound to these two methods.
    '''
def saliency_pearson(cam, occ):
    cam_norm = normalize_abs_0_100(cam).ravel()
    occ_norm = normalize_abs_0_100(occ).ravel()
    return float(np.corrcoef(cam_norm, occ_norm)[0, 1])
