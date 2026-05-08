"""Example script for running Grad-CAM and occlusion on one BreakHis sample.

Edit WEIGHTS_PATH and METADATA_CSV before running.
"""

from whitebox import (
    load_rescnn,
    load_breakhis_sample,
    generate_gradcam,
    show_gradcam_on_image,
    run_occlusion_head1,
    show_occlusion_on_image,
    saliency_difference,
    saliency_pearson,
)

WEIGHTS_PATH = "/content/drive/MyDrive/Semester_7/BDD/ResCNN_best.pth"
METADATA_CSV = "/content/drive/MyDrive/Semester_7/BDD/data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/BreaKHis_metadata.csv"
SAMPLE_INDEX = 166

model = load_rescnn(WEIGHTS_PATH)
sample = load_breakhis_sample(METADATA_CSV, SAMPLE_INDEX)
img_tensor = sample["image_tensor"]
magn_token = sample["magn_token"]

cam, gradcam_target = generate_gradcam(model, img_tensor, magn_token, target_head="head1")
show_gradcam_on_image(img_tensor, cam, title=f"Grad-CAM Head1, target={gradcam_target}")

occ = run_occlusion_head1(model, img_tensor, magn_token, window_hw=32, stride_hw=14, baseline_mode="zero")
show_occlusion_on_image(img_tensor, occ["heat"], title=f"Occlusion Head1, target={occ['target_class']}")

print(f"Grad-CAM target class: {gradcam_target}")
print(f"Occlusion target class: {occ['target_class']}")
print(f"P(target): {occ['p_target']:.4f}")
print(f"Pearson r: {saliency_pearson(cam, occ['heat']):.4f}")

diff = saliency_difference(cam, occ["heat"])
show_occlusion_on_image(img_tensor, diff, title="|Occlusion - Grad-CAM| Difference")
