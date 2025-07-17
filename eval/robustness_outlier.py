import os
import numpy as np
import pandas as pd
import ast
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

score_cols = ['score_if_qnorm', 'score_gmm_qnorm', 'score_cadi_qnorm', 'ensemble_score']
method_names = {'score_if_qnorm': 'IF', 'score_gmm_qnorm': 'GMM', 'score_cadi_qnorm': 'CADI', 'ensemble_score': 'ESEMBLE'}
total_frames = 0
total_anomalous_frames = 0

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
# MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
# OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results_detection")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
SCORE_DIR = "/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv"
MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
# MASKS_DIR = "/home/jlin1/OutlierDetection/UCSDped2/mask"
OUTPUT_DIR = "/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_robustness_curve(df, score_cols, output_dir):
    """Plot robustness curve (AUC/RBDC vs. anomaly score bins)."""
    plt.figure(figsize=(10, 6))
    
    for col in score_cols:
        if col not in df.columns:
            continue

        # Standardize scores per video (optional but recommended)
        df[f"{col}_zscore"] = df.groupby("video_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        
      # Define bins and labels (0.5σ granularity)
        bin_edges = np.arange(0, 1, 0.2).tolist() + [np.inf]  # -3σ to >3σ in 0.5σ steps
        bin_labels = [
            f"{left:.1f}σ to {right:.1f}σ" 
            for left, right in zip(bin_edges[:-1], bin_edges[1:])
        ]
        bin_labels[0] = f"0"  # First bin
        bin_labels[-1] = f"1"  # Last bin

        # Assign bins
        df[f"{col}_bin"] = pd.cut(
            df[f"{col}_zscore"], 
            bins=bin_edges, 
            labels=bin_labels
        )

        # Compute AUC per bin
        bin_aucs = []
        for bin_label in bin_labels:
            bin_data = df[df[f"{col}_bin"] == bin_label]
            if len(bin_data) > 1 and (bin_data["mask_anomaly"].nunique() > 1):
                auc = roc_auc_score(bin_data["mask_anomaly"], bin_data[col])
                bin_aucs.append(auc)
            else:
                bin_aucs.append(np.nan)  # Skip bins with insufficient data

        # Plot the curve
        plt.plot(bin_labels, bin_aucs, marker="o", label=f"{method_names[col]}")

    plt.xlabel("Anomaly Score Bin (Standard Deviations from Mean)")
    plt.ylabel("AUC")
    plt.title("Robustness Curve: AUC vs. Anomaly Score Magnitude")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_curve.png"))
    plt.close()

def plot_roc_pr_curves(df, score_cols, output_dir):
    y_true = df["mask_anomaly"].values

    for col in score_cols:
        if col not in df.columns:
            continue

        y_scores = df[col].values

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{method_names[col]} (AUC = {roc_auc_score(y_true, y_scores):.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {method_names[col]}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"roc_{col}.png"))
        plt.close()

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure()
        plt.plot(recall, precision, label=f"{method_names[col]} (AP = {average_precision_score(y_true, y_scores):.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {method_names[col]}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"pr_{col}.png"))
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
    union = np.count_nonzero(mask)

    return intersection / union if union > 0 else 0.0

def evaluate_soft_scores(df, score_cols):
    print("\n--- Soft Score Evaluation (AUC / AP) ---")
    results = []
    
    for col in score_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in DataFrame")
            continue
            
        auc = roc_auc_score(df['mask_anomaly'], df[col])
        ap = average_precision_score(df['mask_anomaly'], df[col])
        results.append((method_names[col], auc, ap))
        
        # Print individual results
        print(f"{method_names[col]:<5}: AUC = {auc:.4f}, AP = {ap:.4f}")
    
    return pd.DataFrame(results, columns=['Method', 'AUC', 'AP'])

def normalize_per_video(df, score_cols):
    # Group by video and normalize each score column separately
    for col in score_cols:
        if col in df.columns:
            df[col] = df.groupby("video_id")[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
            )
    return df

def compute_rbdc_for_model(df, score_col, iou_thresh, percentile=85):
    df['temp_anomaly'] = threshold_top_percent(df, score_col, percentile)
    tp = ((df['temp_anomaly'] == 1) & (df['iou'] >= iou_thresh) & (df['mask_anomaly'] == 1)).sum()
    fp = ((df['temp_anomaly'] == 1) & ((df['iou'] < iou_thresh) | (df['mask_anomaly'] == 0))).sum()
    fn = ((df['temp_anomaly'] == 0) & (df['mask_anomaly'] == 1)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def compute_rbdc(df, score_cols, iou_thresh=0.0, percentile=85):
    print("\n--- RBDC Evaluation ---")
    rbdc_dict = {}
    for col in score_cols:
        precision, recall, f1 = compute_rbdc_for_model(df, col, iou_thresh, percentile)
        print(f"{method_names[col]}: RBDC Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")
        rbdc_dict[method_names[col]] = precision  # Using method_names to match 'Method' column in metrics_df
    return rbdc_dict


def threshold_top_percent(df, score_col, percentile=85):
    threshold = np.percentile(df[score_col], percentile)
    return (df[score_col] >= threshold).astype(int)

all_results = []
best_method_log = []

scored_csv_path = "/home/jlin1/OutlierDetection/outputs/shanghai_test/alpha_0.9_all_scored.csv"  # Update this name if needed
df = pd.read_csv(scored_csv_path)

df["bbox"] = df.apply(lambda row: [int(row.x1), int(row.y1), int(row.x2), int(row.y2)], axis=1)

# Convert bbox column if needed
if isinstance(df['bbox'].iloc[0], str):
    df['bbox'] = df['bbox'].apply(ast.literal_eval)

all_results = []
best_method_log = []

for video_id in df['frame_dir'].unique():
    video_df = df[df['frame_dir'] == video_id].copy()
    npy_path = os.path.join(MASKS_DIR, f"{video_id}.npy")

    if not os.path.exists(npy_path):
        print(f"Mask file not found for {video_id}")
        continue

    try:
        mask_array = np.load(npy_path)
        total_frames += mask_array.shape[0]
        total_anomalous_frames += np.any(mask_array, axis=(1, 2)).sum()

        results = []
        for frame_idx in range(mask_array.shape[0]):
            frame_mask = mask_array[frame_idx]
            has_mask_anomaly = frame_mask.any()

            frame_objs = video_df[video_df['frame_idx'] == frame_idx]
            IOU_ANOMALY_THRESH = 0.1

            for _, row in frame_objs.iterrows():
                scores = {k: row[k] if k in row else None for k in score_cols}
                iou = compute_mask_iou(row['bbox'], frame_mask)
                is_mask_anomaly = int(iou >= IOU_ANOMALY_THRESH)

                results.append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "track_id": row.get("track_id", None),
                    "bbox": row['bbox'],
                    "mask_anomaly": int(is_mask_anomaly),
                    "iou": iou,
                    **scores
                })

        if results:
            result_df = pd.DataFrame(results)
            all_results.append(result_df)
            output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_iou_comparison.csv")
            result_df.to_csv(output_csv, index=False)

            video_best_metrics = []
            if len(np.unique(result_df['mask_anomaly'])) >= 2:
                for col in score_cols:
                    if col in result_df.columns:
                        auc = roc_auc_score(result_df['mask_anomaly'], result_df[col])
                        ap = average_precision_score(result_df['mask_anomaly'], result_df[col])
                        video_best_metrics.append((col, auc, ap))

            if video_best_metrics:
                best_auc_method = max(video_best_metrics, key=lambda x: x[1])[0]
                best_ap_method = max(video_best_metrics, key=lambda x: x[2])[0]
                best_method_log.append({
                    'video_id': video_id,
                    'best_auc_method': method_names[best_auc_method],
                    'best_ap_method': method_names[best_ap_method]
                })

    except Exception as e:
        print(f"Error processing {video_id}: {e}")
