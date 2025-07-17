import os
import numpy as np

MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"

def summarize_mask_anomalies(masks_dir):
    total_frames = 0
    total_anomalous_frames = 0
    anomaly_distribution = {}

    for filename in os.listdir(masks_dir):
        if not filename.endswith(".npy"):
            continue

        video_id = filename.replace(".npy", "")
        mask_path = os.path.join(masks_dir, filename)

        try:
            mask_array = np.load(mask_path)
            frame_gt = np.any(mask_array, axis=(1, 2)).astype(int)

            num_frames = len(frame_gt)
            num_anomalous = int(frame_gt.sum())

            total_frames += num_frames
            total_anomalous_frames += num_anomalous

            anomaly_distribution[video_id] = {
                "total_frames": num_frames,
                "anomalous_frames": num_anomalous
            }
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return total_frames, total_anomalous_frames, anomaly_distribution

# Run summary
total, anomalous, dist = summarize_mask_anomalies(MASKS_DIR)

print(f"\nTotal frames: {total}")
print(f"Total anomalous frames: {anomalous}")
print("\nAnomaly distribution per video:")
for vid, stats in dist.items():
    print(f"{vid}: {stats['anomalous_frames']} / {stats['total_frames']} anomalous frames")
