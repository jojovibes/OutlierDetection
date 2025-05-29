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

ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
SCORE_DIR = os.path.join(ROOT_DIR, "frames/output")
MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
FRAME_DIR = "/home/jlin1/OutlierDetection/testing/test_frame_mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for score_file in os.listdir(SCORE_DIR):
    if not score_file.endswith("_scored.csv") or score_file.startswith('.'):
        continue

    video_id = score_file.replace("_scored.csv", "")
    csv_path = os.path.join(SCORE_DIR, score_file)
    npy_path = os.path.join(MASKS_DIR, video_id + ".npy")

    if not os.path.exists(npy_path):
        print(f"Mask file not found for {video_id}")
        continue

    try:
        df = pd.read_csv(csv_path)
        mask_array = np.load(npy_path) 

        if 'bbox' not in df.columns or 'frame_idx' not in df.columns or 'cadi_anomaly' not in df.columns:
            print(f"Missing required columns in {score_file}")
            print(f"")
            continue

        if isinstance(df['bbox'].iloc[0], str):
            df['bbox'] = df['bbox'].apply(ast.literal_eval)

        results = []

        for frame_idx in range(mask_array.shape[0]):
            frame_mask = mask_array[frame_idx]
            has_mask_anomaly = frame_mask.any()

            frame_objs = df[df['frame_idx'] == frame_idx]
            # has_score_anomaly = (frame_objs['cadi_anomaly'] == 1).any()
            # print(f"Processing video {video_id}, frame {frame_idx} - Mask anomaly: {has_mask_anomaly}, Score anomaly: {has_score_anomaly}")

            for _, row in frame_objs.iterrows():  
                has_score_anomaly = int(row['cadi_anomaly'] == 1)

                if not has_mask_anomaly and not has_score_anomaly:
                    continue
                
                iou = compute_mask_iou(row['bbox'], frame_mask)

                results.append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "track_id": row.get("track_id", None),
                    "bbox": row['bbox'],
                    "cadi_anomaly": int(row['cadi_anomaly']),
                    "mask_anomaly": int(has_mask_anomaly),
                    "iou": iou
                })

        if results:
            result_df = pd.DataFrame(results)
            output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_iou_comparison.csv")
            result_df.to_csv(output_csv, index=False)
            print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Error processing {score_file}: {e}")


from sklearn.metrics import roc_auc_score, average_precision_score

auc = roc_auc_score(all_results_df['mask_anomaly'], all_results_df['cadi_score'])
ap = average_precision_score(all_results_df['mask_anomaly'], all_results_df['cadi_score'])
precision = precision_score(mask_anomalies, cadi_anomalies)
recall = recall_score(mask_anomalies, cadi_anomalies)
f1 = f1_score(mask_anomalies, cadi_anomalies)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
