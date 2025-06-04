import os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


score_cols = ['score_if', 'score_gmm', 'score_cadi', 'score_avg']
method_names = {'score_if': 'IF', 'score_gmm': 'GMM', 'score_cadi': 'CADI', 'score_avg': 'AVG'}

total_frames = 0
total_anomalous_frames = 0

ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
SCORE_DIR = os.path.join(ROOT_DIR, "small_batch/output")
MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
FRAME_DIR = "/home/jlin1/OutlierDetection/testing/test_frame_mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_confusion_matrix(df, score_col, threshold_percentile=95, output_dir=".", normalize=None):
    threshold = np.percentile(df[score_col], threshold_percentile)
    y_true = df["mask_anomaly"]
    y_pred = (df[score_col] >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title(f"Confusion Matrix: {method_names[score_col]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{score_col}.png"))
    plt.close()



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
    # union = np.logical_or(mask, bbox_mask).sum()
    union = np.count_nonzero(mask)

    return intersection / union if union > 0 else 0.0

def evaluate_soft_scores(df, score_cols):
    print("\n--- Soft Score Evaluation (AUC / AP) ---")
    for col in score_cols:
        auc = roc_auc_score(df['mask_anomaly'], df[col])
        ap = average_precision_score(df['mask_anomaly'], df[col])
        print(f"{method_names[col]}: AUC = {auc:.3f}, AP = {ap:.3f}")

def evaluate_frame_based(frame_df, score_cols):
    print("\n--- Frame-Based Evaluation ---")
    for col in score_cols:
        y_true = frame_df['mask_anomaly']
        y_score = frame_df[col]
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        threshold = np.percentile(y_score, 90)
        y_pred = (y_score >= threshold).astype(int)

        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{method_names[col]}:")
        print(f"  AUC:        {auc:.3f}")
        print(f"  AP:         {ap:.3f}")
        print(f"  Precision:  {p:.3f}")
        print(f"  Recall:     {r:.3f}")
        print(f"  F1 Score:   {f1:.3f}")


def evaluate_binary_scores(df, score_cols, threshold_percentile=90):
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
        save_confusion_matrix(all_results_df, col, threshold_percentile=95, output_dir=OUTPUT_DIR)

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

def compute_rbdc_for_model(df, score_col, iou_thresh, percentile=90):
    df['temp_anomaly'] = threshold_top_percent(df, score_col, percentile)
    tp = ((df['temp_anomaly'] == 1) & (df['iou'] >= iou_thresh) & (df['mask_anomaly'] == 1)).sum()
    fp = ((df['temp_anomaly'] == 1) & ((df['iou'] < iou_thresh) | (df['mask_anomaly'] == 0))).sum()
    fn = ((df['temp_anomaly'] == 0) & (df['mask_anomaly'] == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def compute_rbdc(df, score_cols, iou_thresh=0.0, percentile=90):
    print("\n--- RBDC Evaluation ---")
    for col in score_cols:
        precision, recall, f1 = compute_rbdc_for_model(df, col, iou_thresh, percentile)
        print(f"{method_names[col]}: RBDC Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

def normalize_per_video(df, score_cols):
    return df.groupby("video_id")[score_cols].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))

def threshold_top_percent(df, score_col, percentile=90):
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
        total_frames += mask_array.shape[0]
        total_anomalous_frames += np.any(mask_array, axis=(1, 2)).sum()

        if 'bbox' not in df.columns or 'frame_idx' not in df.columns:
            print(f"Missing required columns in {score_file}")
            continue

        if isinstance(df['bbox'].iloc[0], str):
            df['bbox'] = df['bbox'].apply(ast.literal_eval)

        results = []

        for frame_idx in range(mask_array.shape[0]):
            frame_mask = mask_array[frame_idx]
            has_mask_anomaly = frame_mask.any()
            IOU_ANOMALY_THRESH = 0.1

            frame_objs = df[df['frame_idx'] == frame_idx]

            for _, row in frame_objs.iterrows():
                scores = {k: row[k] if k in row else None for k in score_cols}

                if has_mask_anomaly:
                    iou = compute_mask_iou(row['bbox'], frame_mask)
                else:
                    iou = 0.00
                
                is_mask_anomaly = int(iou >= IOU_ANOMALY_THRESH)

                results.append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "mask_anomaly": int(is_mask_anomaly),
                    "iou": iou,
                    **scores
                })

        if results:
            result_df = pd.DataFrame(results)
            all_results.append(result_df)
            output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_iou_comparison.csv")
            result_df.to_csv(output_csv, index=False)
            # print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Error processing {score_file}: {e}")


if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df[score_cols] = normalize_per_video(all_results_df, score_cols)

    # Sanity check: how many objects overlap a true anomaly mask?
    print("Total detections:", len(all_results_df))
    print("Total with mask_anomaly = 1:", all_results_df['mask_anomaly'].sum())

    print(f"\n--- Ground Truth Frame-Level Stats ---")
    print(f"Total frames across all videos: {total_frames}")
    print(f"Total anomalous frames (pixel-level): {total_anomalous_frames}")

    # Aggregate detection scores to frame level (e.g., max score per frame)
    frame_level_df = all_results_df.groupby(['video_id', 'frame_idx']).agg(
        {**{col: 'max' for col in score_cols}, 'mask_anomaly': 'max'}
    ).reset_index()

    evaluate_soft_scores(all_results_df, score_cols)
    # evaluate_binary_scores(all_results_df, score_cols)
    optimize_thresholds(all_results_df, score_cols)
    compute_rbdc(all_results_df, score_cols)
    evaluate_frame_based(frame_level_df, score_cols)
    plot_score_distributions(all_results_df, score_cols, os.path.join(OUTPUT_DIR, "score_distributions"))
