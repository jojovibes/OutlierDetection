import os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def normalize_per_video(df, score_cols):
    return df.groupby("video_id")[score_cols].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))


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

def compute_rbdc(df, iou_threshold=0.3): #loose detect and threshold adjust
    tp = ((df['cadi_anomaly'] == 1) & (df['iou'] >= iou_threshold) & (df['mask_anomaly'] == 1)).sum()
    fp = ((df['cadi_anomaly'] == 1) & ((df['iou'] < iou_threshold) | (df['mask_anomaly'] == 0))).sum()
    fn = ((df['cadi_anomaly'] == 0) & (df['mask_anomaly'] == 1)).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1

def threshold_top_percent(df, score_col, percentile=95):
    threshold = np.percentile(df[score_col], percentile)
    return (df[score_col] >= threshold).astype(int)

ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
SCORE_DIR = os.path.join(ROOT_DIR, "frames/output")
MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
FRAME_DIR = "/home/jlin1/OutlierDetection/testing/test_frame_mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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




if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)

    score_cols = ['score_if', 'score_gmm', 'score_cadi', 'score_avg']

    all_results_df[score_cols] = normalize_per_video(all_results_df, ['score_if', 'score_gmm', 'score_cadi', 'score_avg'])

    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    for method in score_cols:
        if method in all_results_df.columns:
            plt.figure(figsize=(8, 4))
            plt.hist(all_results_df[all_results_df['mask_anomaly'] == 1][method], bins=50, alpha=0.5, label="Anomaly")
            plt.hist(all_results_df[all_results_df['mask_anomaly'] == 0][method], bins=50, alpha=0.5, label="Normal")
            plt.title(f"Score Distribution: {method}")
            plt.xlabel("Normalized Score")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"{method}_distribution.png"))
            plt.close()

    # # Normalize scores
    # score_cols = ['score_if', 'score_gmm', 'score_cadi', 'score_avg']
    # scaler = MinMaxScaler()
    # for col in score_cols:
    #     if col in all_results_df.columns:
    #         all_results_df[col] = scaler.fit_transform(all_results_df[[col]])

    print("\n--- Soft Score Evaluation ---")
    for method in ['if', 'gmm', 'cadi', 'avg']:
        score_col = f"score_{method}"
        if score_col in all_results_df.columns:
            auc = roc_auc_score(all_results_df['mask_anomaly'], all_results_df[score_col])
            ap = average_precision_score(all_results_df['mask_anomaly'], all_results_df[score_col])
            print(f"\n{method.upper()}:")
            print(f"AUC: {auc:.3f}")
            print(f"AP:  {ap:.3f}")

    print("\n--- Binary Evaluation (Top 5% Threshold) ---")
    for method in ['if', 'gmm', 'cadi', 'avg']:
        score_col = f"score_{method}"
        if score_col in all_results_df.columns:
            pred_bin = threshold_top_percent(all_results_df, score_col)
            p = precision_score(all_results_df['mask_anomaly'], pred_bin)
            r = recall_score(all_results_df['mask_anomaly'], pred_bin)
            f = f1_score(all_results_df['mask_anomaly'], pred_bin)
            print(f"\n{method.upper()}:")
            print(f"Precision: {p:.3f}")
            print(f"Recall:    {r:.3f}")
            print(f"F1 Score:  {f:.3f}")

    # RBDC using CADI
    if 'cadi_score' in all_results_df.columns:
        all_results_df['cadi_anomaly'] = threshold_top_percent(all_results_df, 'cadi_score')
        rbdc_p, rbdc_r, rbdc_f1 = compute_rbdc(all_results_df)
        print(f"\n--- RBDC Evaluation (CADI only, IoU >= 0.5) ---")
        print(f"RBDC Precision: {rbdc_p:.3f}")
        print(f"RBDC Recall:    {rbdc_r:.3f}")
        print(f"RBDC F1 Score:  {rbdc_f1:.3f}")
else:
    print("No results to evaluate.")


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

print("\n--- Optimal Thresholds (Max F1) ---")
for method in ['if', 'gmm', 'cadi', 'avg']:
    score_col = f"score_{method}"
    if score_col in all_results_df.columns:
        best_thresh, best_f1 = optimize_threshold(all_results_df, score_col)
        print(f"{method.upper()}: Best threshold = {best_thresh:.3f}, Max F1 = {best_f1:.3f}")

for method in ['score_if', 'score_gmm', 'score_cadi', 'score_avg']:
    if method in all_results_df.columns:
        print(f"{method}: min={all_results_df[method].min():.3f}, max={all_results_df[method].max():.3f}, mean={all_results_df[method].mean():.3f}")
