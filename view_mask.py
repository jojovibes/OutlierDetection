import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
mask = np.load("/Volumes/ronni/shanghaitech/testing/test_pixel_mask/01_0016.npy")

# Show basic info
print("Mask shape:", mask.shape)
print("Unique values:", np.unique(mask))

# # Display the mask visually
# plt.imshow(mask, cmap='gray')
# plt.title("Pixel-level Anomaly Mask")
# plt.colorbar()
# plt.show()

step = 10  # change this to view more/less frequently

for frame_idx in range(0, mask.shape[0], step):
    frame_mask = mask[frame_idx]

    plt.figure(figsize=(8, 4))
    plt.imshow(frame_mask, cmap='gray', vmin=0, vmax=1)  # binary contrast
    plt.title(f"Anomaly Mask - Frame {frame_idx}")
    plt.axis('off')
    plt.show()