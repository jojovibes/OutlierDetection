import os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

score_cols = ['score_if', 'score_gmm', 'score_cadi', 'score_avg']
method_names = {'score_if': 'IF', 'score_gmm': 'GMM', 'score_cadi': 'CADI', 'score_avg': 'AVG'}

ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
SCORE_DIR = os.path.join(ROOT_DIR, "small_batch/output")
MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
FRAME_DIR = "/home/jlin1/OutlierDetection/testing/test_frame_mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

def evaluate_soft_scores(df, score_cols):
    print("\n--- Soft Score Evaluation (AUC / AP) ---")
    for col in score_cols:
        auc = roc_auc_score(df['mask_anomaly'], df[col])
        ap = average_precision_score(df['mask_anomaly'], df[col])
        print(f"{method_names[col]}: AUC = {auc:.3f}, AP = {ap:.3f}")

def evaluate_binary_scores(df, score_cols, threshold_percentile=95):
    print(f"\n--- Binary Evaluation (Top {100 - threshold_percentile}% Threshold) ---")
    for col in score_cols:
        preds = threshold_top_percent(df, col, threshold_percentile)
        p = precision_score(df['mask_anomaly'], preds)
        r = recall_score(df['mask_anomaly'], preds)
        f1 = f1_score(df['mask_anomaly'], preds)
        print(f"{method_names[col]}: Precision = {p:.3f}, Recall = {r:.3f}, F1 = {f1:.3f}")

def optimize_thresholds(df, score_cols):
    print("\n--- Optimal Thresholds (Max F1) ---")
    for col in score_cols:
        best_thresh, best_f1 = optimize_threshold(df, col)
        print(f"{method_names[col]}: Best threshold = {best_thresh:.3f}, Max F1 = {best_f1:.3f}")

def plot_score_distributions(df, score_cols, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    for col in score_cols:
        plt.figure(figsize=(8, 4))
        plt.hist(df[df['mask_anomaly'] == 1][col], bins=50, alpha=0.5, label="Anomaly")
        plt.hist(df[df['mask_anomaly'] == 0][col], bins=50, alpha=0.5, label="Normal")
        plt.title(f"Score Distribution: {method_names[col]}")
        plt.xlabel("Normalized Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{col}_distribution.png"))
        plt.close()

# def compute_rbdc_for_model(df, score_col, iou_thresh=0.3, percentile=95):
#     df['temp_anomaly'] = threshold_top_percent(df, score_col, percentile)
#     tp = ((df['temp_anomaly'] == 1) & (df['iou'] >= iou_thresh) & (df['mask_anomaly'] == 1)).sum()
#     fp = ((df['temp_anomaly'] == 1) & ((df['iou'] < iou_thresh) | (df['mask_anomaly'] == 0))).sum()
#     fn = ((df['temp_anomaly'] == 0) & (df['mask_anomaly'] == 1)).sum()
#     precision = tp / (tp + fp + 1e-10)
#     recall = tp / (tp + fn + 1e-10)
#     f1 = 2 * precision * recall / (precision + recall + 1e-10)
#     return precision, recall, f1

def compute_rbdc_for_model(df, score_col, iou_thresh, percentile=95):
    df['temp_anomaly'] = threshold_top_percent(df, score_col, percentile)
    tp = ((df['temp_anomaly'] == 1) & (df['iou'] >= iou_thresh) & (df['mask_anomaly'] == 1)).sum()
    fp = ((df['temp_anomaly'] == 1) & ((df['iou'] < iou_thresh) | (df['mask_anomaly'] == 0))).sum()
    fn = ((df['temp_anomaly'] == 0) & (df['mask_anomaly'] == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def compute_rbdc(df, score_cols, iou_thresh=0.0, percentile=95):
    print("\n--- RBDC Evaluation ---")
    for col in score_cols:
        precision, recall, f1 = compute_rbdc_for_model(df, col, iou_thresh, percentile)
        print(f"{method_names[col]}: RBDC Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

def normalize_per_video(df, score_cols):
    return df.groupby("video_id")[score_cols].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))

def threshold_top_percent(df, score_col, percentile=95):
    threshold = np.percentile(df[score_col], percentile)
    return (df[score_col] >= threshold).astype(int)

def optimize_threshold(df, score_col):
    best_thresh = 0
    best_f1 = 0
    thresholds = np.linspace(0, 1, 100)

    for thresh in thresholds:
        preds = (df[score_col] >= thresh).astype(int)
        f1 = f1_score(df['mask_anomaly'], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1

def optimize_thresholds(df, score_cols):
    print("\n--- Optimal Thresholds (Max F1) ---")
    for col in score_cols:
        best_thresh, best_f1 = optimize_threshold(df, col)
        print(f"{method_names[col]}: Best threshold = {best_thresh:.3f}, Max F1 = {best_f1:.3f}")


all_results = []

# Loop through all scored CSVs
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

        if 'bbox' not in df.columns or 'frame_idx' not in df.columns:
            print(f"Missing required columns in {score_file}")
            continue

        if isinstance(df['bbox'].iloc[0], str):
            df['bbox'] = df['bbox'].apply(ast.literal_eval)

        results = []

        for frame_idx in range(mask_array.shape[0]):
            frame_mask = mask_array[frame_idx]
            has_mask_anomaly = frame_mask.any()

            frame_objs = df[df['frame_idx'] == frame_idx]

            for _, row in frame_objs.iterrows():
                scores = {k: row[k] if k in row else None for k in ['score_if', 'score_gmm', 'score_cadi', 'score_avg']}
                iou = compute_mask_iou(row['bbox'], frame_mask)
                results.append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "track_id": row.get("track_id", None),
                    "bbox": row['bbox'],
                    "mask_anomaly": int(has_mask_anomaly),
                    "iou": iou,
                    **scores
                })

        if results:
            result_df = pd.DataFrame(results)
            all_results.append(result_df)
            output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_iou_comparison.csv")
            result_df.to_csv(output_csv, index=False)
            print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Error processing {score_file}: {e}")

print(all_results)

if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)

    evaluate_soft_scores(all_results_df, score_cols)
    evaluate_binary_scores(all_results_df, score_cols)
    optimize_thresholds(all_results_df, score_cols)
    compute_rbdc(all_results_df, score_cols)