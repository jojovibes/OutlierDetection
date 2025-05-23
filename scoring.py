import os
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# scaler = MinMaxScaler()
# normalized_scores = scaler.fit_transform(scores.reshape(-1, 1))


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


ROOT_DIR = "/path/to/dataset" 
OUTPUT_DIR = os.path.join(ROOT_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

final`Ω = []

for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    csv_path = os.path.join(folder_path, f"{folder}_features.csv")
    mask_path = os.path.join(folder_path, f"{folder}.npy")

    if not os.path.exists(csv_path) or not os.path.exists(mask_path):
        print(f"Missing CSV or mask for {folder}")
        continue

    print(f"Processing: {folder}")

    df = pd.read_csv(csv_path)

    if isinstance(df['bbox'].iloc[0], str):
        df['bbox'] = df['bbox'].apply(ast.literal_eval) # Convert string to list

    mask = np.load(mask_path)
    results = []

    for frame_idx in range(mask.shape[0]):
        frame_mask = mask[frame_idx]
        has_mask_anomaly = frame_mask.any()

        frame_objs = df[df['frame_idx'] == frame_idx]
        has_cadi_anomaly = frame_objs['cadi_anomaly'].sum() > 0

        if has_mask_anomaly or has_cadi_anomaly:
            for _, row in frame_objs.iterrows():
                bbox = row['bbox']
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

    out_df = pd.DataFrame(results)
    all_results.append(out_df)
    out_path = os.path.join(OUTPUT_DIR, f"{folder}_iou_comparison.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

final_df = pd.concat(all_results, ignore_index=True)


# Load the aggregated results
mask_anomalies = final_df['mask_anomaly']
cadi_anomalies = final_df['cadi_anomaly']

from sklearn.metrics import roc_auc_score, average_precision_score

auc = roc_auc_score(all_results_df['mask_anomaly'], all_results_df['cadi_score'])
ap = average_precision_score(all_results_df['mask_anomaly'], all_results_df['cadi_score'])
precision = precision_score(mask_anomalies, cadi_anomalies)
recall = recall_score(mask_anomalies, cadi_anomalies)
f1 = f1_score(mask_anomalies, cadi_anomalies)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
