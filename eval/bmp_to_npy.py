import os
import numpy as np
import cv2

src_root = "/home/jlin1/OutlierDetection/UCSDped2/Test"       # contains both TestXXX and TestXXX_gt folders
out_root = "/home/jlin1/OutlierDetection/UCSDped2/mask"           
os.makedirs(out_root, exist_ok=True)

for folder in sorted(os.listdir(src_root)):
    if not folder.startswith("Test") or not folder.endswith("_gt"):
        continue

    video_id = folder.replace("_gt", "")
    gt_dir = os.path.join(src_root, folder)

    bmp_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".bmp")])
    if not bmp_files:
        print(f"Warning: no .bmp files in {folder}")
        continue

    # Get mask shape from first frame
    sample = cv2.imread(os.path.join(gt_dir, bmp_files[0]), 0)
    H, W = sample.shape
    T = len(bmp_files)

    masks = np.zeros((T, H, W), dtype=np.uint8)

    for idx, fname in enumerate(bmp_files):
        bmp_path = os.path.join(gt_dir, fname)
        bmp = cv2.imread(bmp_path, 0)
        masks[idx] = (bmp > 0).astype(np.uint8)

    out_path = os.path.join(out_root, f"{video_id}.npy")
    np.save(out_path, masks)
    print(f"Saved {video_id}.npy: shape = {masks.shape}")
