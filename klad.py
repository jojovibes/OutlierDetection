import os
import numpy as np
import pandas as pd
import ast

def compute_mask_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2+1, x1:x2+1] = True

    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.logical_or(mask, bbox_mask).sum()

    return intersection / union if union > 0 else 0.0

def compare_score_and_mask(score_path, mask_path):
   print(f"Comparing {score_path} with {mask_path}")

   df = pd.read_csv(score_path)
    
   results = []
   mask = np.load(mask_path)

   for frame_idx in range(mask.shape[0]):
        frame_mask = mask[frame_idx]
        has_mask_anomaly = frame_mask.any()

        frame_objs = df[df['frame_idx'] == frame_idx]
        has_cadi_anomaly = frame_objs['cadi_anomaly'].sum() > 0
        # has_score_anomaly = (frame_objs['cadi_anomaly'] == 1).any()

        if has_mask_anomaly or has_cadi_anomaly:
            for _, row in frame_objs.iterrows():

                # bbox = row['bbox']
                w = row['width']
                h = row['height']
                cx = row['center_x']
                cy = row['center_y']
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                bbox = [x1, y1, x2, y2]

                is_cadi_anomaly = row['cadi_anomaly'] == 1
                iou = compute_mask_iou(bbox, frame_mask)

                results.append({
                    "frame_idx": frame_idx,
                    "track_id": row.get("track_id", None),
                    "bbox": bbox,
                    "cadi_anomaly": int(is_cadi_anomaly),
                    "mask_anomaly": int(iou > 0),
                    "iou": iou
                })

        result_df = pd.DataFrame(results)
        # print(f"Compared {len(result_df)} detections from {score_path} to {mask_path}")
        return result_df
          

ROOT_DIR = "/Volumes/ronni/shanghaitech/testing/small_batch/output/scores" 
MASK_DIR = os.path.join("/Volumes/ronni/shanghaitech/testing/test_pixel_mask")
OUTPUT_DIR = os.path.join("/Volumes/ronni/shanghaitech/testing/small_batch/output/comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

matches = []

for mask in os.listdir(MASK_DIR):
    if mask.startswith('.'): 
        continue
    base = os.path.splitext(mask)[0] 
    for score in os.listdir(ROOT_DIR):
        if score.startswith('.'):  
            continue
        if base in score:
            matches.append((score, mask))
            break  # Only one match 

for score, mask in matches:
    # print(f"Matched: {score} <--> {mask}")
    score_path = os.path.join(ROOT_DIR, score)
    mask_path = os.path.join(MASK_DIR, mask)
    result_df = compare_score_and_mask(score_path, mask_path)

    out_path = os.path.join(OUTPUT_DIR, base + "_iou_comparison.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")




# for folder in os.listdir(ROOT_DIR):
#     folder_path = os.path.join(ROOT_DIR, folder)

#     if not os.path.isdir(folder_path):
#         continue

#     csv_path = os.path.join(folder_path, f"{folder}_features.csv")
#     mask_path = os.path.join(folder_path, f"{folder}.npy")

#     if not os.path.exists(csv_path) or not os.path.exists(mask_path):
#         print(f"Missing CSV or mask for {folder}")
#         continue

#     print(f"Processing: {folder}")

#     df = pd.read_csv(csv_path)

#     if isinstance(df['bbox'].iloc[0], str):
#         df['bbox'] = df['bbox'].apply(ast.literal_eval) # Convert string to list

#     mask = np.load(mask_path)
#     results = []

#     for frame_idx in range(mask.shape[0]):
#         frame_mask = mask[frame_idx]
#         has_mask_anomaly = frame_mask.any()

#         frame_objs = df[df['frame_idx'] == frame_idx]
#         has_cadi_anomaly = frame_objs['cadi_anomaly'].sum() > 0

#         if has_mask_anomaly or has_cadi_anomaly:
#             for _, row in frame_objs.iterrows():
#                 bbox = row['bbox']
#                 is_cadi_anomaly = row['cadi_anomaly'] == 1
#                 iou = compute_mask_iou(bbox, frame_mask)

#                 results.append({
#                     "frame_idx": frame_idx,
#                     "track_id": row.get("track_id", None),
#                     "bbox": bbox,
#                     "cadi_anomaly": int(is_cadi_anomaly),
#                     "mask_anomaly": int(iou > 0),
#                     "iou": iou
#                 })

#     out_df = pd.DataFrame(results)
#     out_path = os.path.join(OUTPUT_DIR, f"{folder}_iou_comparison.csv")
#     out_df.to_csv(out_path, index=False)


#     out_name = os.path.splitext(score)[0] + "_iou_comparison.csv"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     result_df.to_csv(out_path, index=False)
#     print(f"Saved: {out_path}")
