# White-box Explainability Codebase

This folder contains a cleaned modular version of the Grad-CAM and occlusion code and the.

## Structure

```text
Whitebox/
    whitebox/
        model.py        # ResCNN architecture and load_rescnn()
        gradcam.py      # Grad-CAM generation and display
        occlusion.py    # Occlusion Δp generation and display
        data.py         # BreakHis sample loading and magnification-token   conversion
        analysis.py     # Difference map and Pearson correlation helpers
    scripts/
      run_single_image.py
    requirements.txt
    README.md
```

## Notebook usage

All code is currently set up to be run within Google Colab.  As with the rest of this project, you will need to have the correctly organized file structure and adjust the paths from mine as needed.  But, generally, these steps should get you going:

```python
!pip install -r /content/drive/MyDrive/Colab\ Notebooks/Whitebox/requirements.txt
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/Whitebox')

from whitebox import (
    load_rescnn, load_breakhis_sample,
    generate_gradcam, show_gradcam_on_image,
    run_occlusion_head1, show_occlusion_on_image,
    saliency_difference, saliency_pearson,
)
```

Then:

```python
weights_path = "/content/drive/MyDrive/Semester_7/BDD/ResCNN_best.pth"
metadata_csv = "/content/drive/MyDrive/Semester_7/BDD/data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/BreaKHis_metadata.csv"

model = load_rescnn(weights_path)
sample = load_breakhis_sample(metadata_csv, index=166)
img_tensor = sample["image_tensor"]
magn_token = sample["magn_token"]

cam, target_class = generate_gradcam(model, img_tensor, magn_token, target_head="head1")
show_gradcam_on_image(img_tensor, cam, title=f"Grad-CAM Head1, target={target_class}")

occ = run_occlusion_head1(model, img_tensor, magn_token, window_hw=32, stride_hw=14, baseline_mode="zero")
show_occlusion_on_image(img_tensor, occ["heat"], title=f"Occlusion Head1, target={occ['target_class']}")

print("Pearson r:", saliency_pearson(cam, occ["heat"]))
diff = saliency_difference(cam, occ["heat"])
show_occlusion_on_image(img_tensor, diff, title="|Occlusion - Grad-CAM| Difference")
```

You can do this example in run_single_image.py.  There is also Image_Run_Example.ipynb to the same end. 


For questions, contact: Jonathan Brown <jbrow365@jh.edu>
