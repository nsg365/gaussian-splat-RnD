import cv2
import numpy as np
import os
from tqdm import tqdm

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
TARGET_FILE = "dataset/targets.txt"

TOP_K_RESULTS = 5
MAX_SPLATS = 300
SIGMA_CORR = 30.0   # correspondence bandwidth (pixels)

# LOAD FILE LISTS
with open(TARGET_FILE, "r") as f:
    targets = [l.strip() for l in f if l.strip()]

all_images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

sift = cv2.SIFT_create()

# GAUSSIAN SPLAT EXTRACTION

def extract_gaussian_splats(image, mask=None, max_splats=MAX_SPLATS):
    """
    Returns list of splats:
    (x, y, sigma, weight)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray, None)

    splats = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Mask-aware filtering
        if mask is not None and mask[y, x] == 255:
            continue

        sigma = kp.size / 2.0
        weight = kp.response
        splats.append((x, y, sigma, weight))

    # Saliency-based pruning
    splats.sort(key=lambda s: s[3], reverse=True)
    return splats[:max_splats]

# SOFT GAUSSIAN ENERGY (MINIMIZATION)

def soft_gaussian_energy(A, B, sigma=SIGMA_CORR):
    """
    Computes energy between splat sets A and B
    using soft Gaussian correspondences.
    Lower energy = better match.
    """
    if len(A) == 0 or len(B) == 0:
        return np.inf

    XA = np.array([[x, y] for x, y, _, _ in A])
    XB = np.array([[x, y] for x, y, _, _ in B])

    # Pairwise squared distances
    D2 = np.sum((XA[:, None, :] - XB[None, :, :])**2, axis=2)

    # Soft correspondence matrix
    P = np.exp(-D2 / (2 * sigma**2))
    P /= np.sum(P, axis=1, keepdims=True) + 1e-8

    # Expected squared distance (energy)
    energy = np.sum(P * D2)
    return energy

# PRECOMPUTE SOURCE SPLATS
print("Extracting splats for source images...")
source_splats = {}

for img_name in tqdm(all_images):
    img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    if img is None:
        continue
    source_splats[img_name] = extract_gaussian_splats(img)

# RETRIEVAL 
for target in targets:
    print(f"\n=== Target: {target} ===")

    img_path = os.path.join(IMAGE_DIR, target)
    mask_path = os.path.join(
        MASK_DIR, target.rsplit(".", 1)[0] + "_mask.png"
    )

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if image is None or mask is None:
        print("Missing image or mask, skipping.")
        continue

    target_splats = extract_gaussian_splats(image, mask)

    energies = []
    for src_name, src_splats in source_splats.items():
        if src_name == target:
            continue

        energy = soft_gaussian_energy(target_splats, src_splats)
        energies.append((src_name, energy))

    energies.sort(key=lambda x: x[1])

    print("Top retrieved images (min energy):")
    for name, energy in energies[:TOP_K_RESULTS]:
        print(f"  {name} | energy = {energy:.4f}")
