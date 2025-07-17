import os
import numpy as np
from scipy.io import loadmat

# Path to the CUHK Avenue ground-truth .mat file
input_dir = "/home/jlin1/OutlierDetection/Avenue/masks"
output_dir = "/home/jlin1/OutlierDetection/Avenue/masks_npy"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".mat"):
        continue

    mat_path = os.path.join(input_dir, fname)
    mat_data = loadmat(mat_path)

    # Search for a 3D anomaly mask inside each file
    binary_mask = None
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            binary_mask = (value > 0).astype(np.uint8)
            break

    if binary_mask is None:
        print(f"Skipped {fname}: no 3D binary mask found")
        continue

    npy_name = os.path.splitext(fname)[0] + ".npy"
    np.save(os.path.join(output_dir, npy_name), binary_mask)
    print(f"Saved {npy_name}: shape {binary_mask.shape}")