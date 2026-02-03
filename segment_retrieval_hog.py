import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndimage
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks1"             
SEGMENT_BASE_DIR = "dataset/intra_segments" 
TARGET_FILE = "dataset/targets.txt"

TOP_K = 5
MAX_SPLATS_TARGET = 300  # Max blobs to track on the target
MAX_SPLATS_SEGMENT = 100 # Segments are small, so we need fewer blobs
MAX_DIST = 50

# --- 1. HESSIAN OF GAUSSIAN (HoG) DETECTOR ---
def detect_hessian_blobs(image_gray, mask=None, threshold=2000):
    splats = []
    # Segments are small, so we focus on smaller scales
    scales = [1.0, 3.0, 5.0] 
    
    for sigma in scales:
        k = int(6 * sigma) | 1 
        if k < 3: k = 3
        
        blurred = cv2.GaussianBlur(image_gray, (k, k), sigma)
        Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
        
        det_H = (Ixx * Iyy) - (Ixy ** 2)
        local_max = ndimage.maximum_filter(det_H, size=5)
        peaks = (det_H == local_max) & (det_H > threshold)
        
        y_coords, x_coords = np.where(peaks)
        
        for y, x in zip(y_coords, x_coords):
            if mask is not None and mask[y, x] == 255:
                continue
            weight = det_H[y, x]
            splats.append((int(x), int(y), sigma, float(weight)))
    return splats

def extract_hog_features(image, mask=None, max_splats=100):
    if image is None: return []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    raw_splats = detect_hessian_blobs(gray, mask, threshold=3000)
    raw_splats.sort(key=lambda s: s[3], reverse=True)
    return raw_splats[:max_splats]

# --- 2. SIMILARITY SCORE ---
def splat_similarity(A, B):
    # A = Target Splats, B = Segment Splats
    score = 0.0
    for xa, ya, sa, wa in A:
        best_match = 0.0
        for xb, yb, sb, wb in B:
            # We relax the distance constraint because segments can be anywhere
            # But we check if the "scale" (sigma) matches
            if abs(sa - sb) > 2.0: 
                continue
                
            # Similarity based on weight (sharpness/texture)
            match_val = min(wa, wb)
            best_match = max(best_match, match_val)
        score += best_match
    return score

# --- 3. DATABASE LOADER (WITH PROGRESS BAR) ---
def load_segment_database():
    print(f"Scanning files in {SEGMENT_BASE_DIR}...")
    
    if not os.path.exists(SEGMENT_BASE_DIR):
        print("Error: Segment folder not found. Did you run image_split.py?")
        return {}, {}

    # 1. First, just collect all file paths (fast)
    all_segment_files = []
    for root, dirs, files in os.walk(SEGMENT_BASE_DIR):
        for file in files:
            if file.startswith("region_") and file.endswith(".png"):
                all_segment_files.append(os.path.join(root, file))

    print(f"Found {len(all_segment_files)} segments. Extracting features...")

    db_splats = {}
    db_images = {} 
    
    # 2. Now process them with a progress bar
    for full_path in tqdm(all_segment_files):
        # Create ID: "ImageName_RegionID"
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
        file_name = os.path.basename(full_path)
        seg_id = f"{parent_folder}_{file_name}"
        
        img = cv2.imread(full_path)
        if img is not None:
            # Extract features
            feats = extract_hog_features(img, max_splats=MAX_SPLATS_SEGMENT)
            if len(feats) > 2: # Only keep segments with actual details
                db_splats[seg_id] = feats
                db_images[seg_id] = img
                        
    print(f"Successfully loaded {len(db_splats)} useful segments.")
    return db_splats, db_images

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load Data
    segment_feats, segment_imgs = load_segment_database()
    
    with open(TARGET_FILE, "r") as f:
        targets = [l.strip() for l in f if l.strip()]

    # Process each target
    for target_name in targets:
        print(f"\nProcessing Target: {target_name}...")
        
        img_path = os.path.join(IMAGE_DIR, target_name)
        mask_path = os.path.join(MASK_DIR, target_name.rsplit(".",1)[0] + "_mask.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print("  Missing image or mask.")
            continue
            
        target_img = cv2.imread(img_path)
        target_mask = cv2.imread(mask_path, 0)
        
        # Extract features from target (Target has more features than segments)
        target_splats = extract_hog_features(target_img, target_mask, max_splats=MAX_SPLATS_TARGET)
        
        # Compare against all segments
        scores = []
        for seg_id, seg_splats in segment_feats.items():
            # Don't match the target image's own segments against itself
            if target_name.rsplit(".",1)[0] in seg_id:
                continue
                
            score = splat_similarity(target_splats, seg_splats)
            scores.append((seg_id, score))
            
        # Sort and Normalize
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores:
            print("  No matches found.")
            continue

        max_score = scores[0][1] if scores[0][1] > 0 else 1.0
        top_results = scores[:TOP_K]

        # --- VISUALIZATION ---
        fig, axes = plt.subplots(1, TOP_K + 1, figsize=(18, 5))
        
        # 1. Target with Mask
        overlay = target_img.copy()
        if target_mask is not None:
            overlay[target_mask == 255] = (0.3 * overlay[target_mask == 255] + 0.7 * np.array([255,0,0])).astype(np.uint8)
            
        axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Target\n{target_name}")
        axes[0].axis("off")
        
        # 2. Top Matching Segments
        for i, (seg_id, raw_score) in enumerate(top_results):
            norm_score = (raw_score / max_score) * 100.0
            seg_img = segment_imgs[seg_id]
            
            axes[i+1].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Match {i+1}\n{seg_id}\nScore: {norm_score:.1f}")
            axes[i+1].axis("off")
            
        plt.tight_layout()
        plt.show()