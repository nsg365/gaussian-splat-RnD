import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
TARGET_FILE = "dataset/targets.txt"
TOP_K = 5

# LOAD RETRIEVAL RESULTS

from tqdm import tqdm

MAX_SPLATS = 300
MAX_DIST = 50

sift = cv2.SIFT_create()

def extract_gaussian_splats(image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray, None)

    splats = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if mask is not None and mask[y, x] == 255:
            continue
        sigma = kp.size / 2.0
        splats.append((x, y, sigma, kp.response))

    splats.sort(key=lambda s: s[3], reverse=True)
    return splats[:MAX_SPLATS]

def splat_similarity(A, B):
    score = 0.0
    for xa, ya, sa, wa in A:
        best = 0.0
        for xb, yb, sb, wb in B:
            if abs(xa - xb) > MAX_DIST or abs(ya - yb) > MAX_DIST:
                continue
            d2 = (xa - xb)**2 + (ya - yb)**2
            sigma = sa + sb + 1e-6
            val = np.exp(-d2 / (2 * sigma * sigma)) * wa * wb
            best = max(best, val)
        score += best
    return score

# LOAD DATA
with open(TARGET_FILE, "r") as f:
    targets = [l.strip() for l in f if l.strip()]

all_images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Precompute source splats
print("Precomputing source splats...")
source_splats = {}
for name in tqdm(all_images):
    img = cv2.imread(os.path.join(IMAGE_DIR, name))
    if img is not None:
        source_splats[name] = extract_gaussian_splats(img)


# VISUALIZATION
for target in targets[:5]:   # visualize only first 5 targets
    img = cv2.imread(os.path.join(IMAGE_DIR, target))
    mask = cv2.imread(
        os.path.join(MASK_DIR, target.rsplit(".",1)[0] + "_mask.png"), 0
    )

    target_splats = extract_gaussian_splats(img, mask)

    scores = []
    for src_name, src_splats in source_splats.items():
        if src_name == target:
            continue
        score = splat_similarity(target_splats, src_splats)
        scores.append((src_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_imgs = scores[:TOP_K]

    # Plot
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(18, 4))

    # Target + mask overlay
    overlay = img.copy()
    overlay[mask == 255] = (0.3 * overlay[mask == 255] + 0.7 * np.array([255,0,0])).astype(np.uint8)
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Target + Mask")
    axes[0].axis("off")

    for i, (name, score) in enumerate(top_imgs):
        simg = cv2.imread(os.path.join(IMAGE_DIR, name))
        axes[i+1].imshow(cv2.cvtColor(simg, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"{name}\nscore={score:.3f}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.show()
