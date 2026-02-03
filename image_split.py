import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2lab

IMAGE_DIR = "dataset/images"
OUT_DIR = "dataset/intra_segments"
NUM_SEGMENTS = 40
COMPACTNESS = 35
MIN_REGION_AREA = 5000

os.makedirs(OUT_DIR, exist_ok=True)

def segment_image(image_path, out_base):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    # SLIC segmentation
    segments = slic(
        img_float,
        n_segments=NUM_SEGMENTS,
        compactness=COMPACTNESS,
        start_label=0
    )

    h, w = segments.shape
    unique_segments = np.unique(segments)

    seg_dir = os.path.join(out_base, "regions")
    mask_dir = os.path.join(out_base, "masks")

    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for seg_id in unique_segments:
        mask = (segments == seg_id).astype(np.uint8)

        if np.sum(mask) < MIN_REGION_AREA:
            continue

        # save mask
        mask_path = os.path.join(mask_dir, f"mask_{seg_id}.png")
        cv2.imwrite(mask_path, mask * 255)

        # extract region
        region = img.copy()
        region[mask == 0] = 0

        region_path = os.path.join(seg_dir, f"region_{seg_id}.png")
        cv2.imwrite(region_path, region)

def main():
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        out_base = os.path.join(OUT_DIR, base_name)

        print(f"[+] Segmenting {img_name}")
        segment_image(img_path, out_base)

if __name__ == "__main__":
    main()