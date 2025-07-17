import os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score)
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

score_cols = ['score_if', 'score_gmm', 'score_cadi', 'score_avg']
method_names = {'score_if': 'IF', 'score_gmm': 'GMM', 'score_cadi': 'CADI', 'score_avg': 'AVG'}
total_frames = 0
total_anomalous_frames = 0

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
# SCORE_DIR = "/home/jlin1/OutlierDetection/testing/small_batch/preprocess_fix"
SCORE_DIR = "/home/jlin1/OutlierDetection/UCSDped2/frames/preprocess_fix"
# MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
MASKS_DIR = "/home/jlin1/OutlierDetection/UCSDped2/mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_roc_pr_curves_overlay(df, score_cols, output_dir):
    y_true = df["mask_anomaly"].values

    # ROC overlay
    plt.figure(figsize=(8, 6))
    for col in score_cols:
        if col not in df.columns:
            continue
        y_scores = df[col].values
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"{method_names[col]} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overlayed ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlayed_roc_curve.png"))
    plt.close()

    # PR overlay
    plt.figure(figsize=(8, 6))
    for col in score_cols:
        if col not in df.columns:
            continue
        y_scores = df[col].values
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f"{method_names[col]} (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overlayed Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlayed_pr_curve.png"))
    plt.close()

def plot_auc_ap_bar(metrics_df, output_dir):
    plt.figure(figsize=(8, 5))
    df_melted = metrics_df.melt(id_vars="Method", value_vars=["AUC", "AP"], var_name="Metric", value_name="Score")
    sns.barplot(x="Method", y="Score", hue="Metric", data=df_melted)
    plt.title("AUC and Average Precision per Method")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auc_ap_barplot.png"))
    plt.close()

# Plot correlation heatmap
def plot_score_correlation(df, score_cols, output_dir):
    corr = df[score_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={'label': 'Correlation'})
    plt.title("Score Correlation Between Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_correlation_heatmap.png"))
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

all_results = []
best_method_log = []

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

            frame_objs = df[df['frame_idx'] == frame_idx]
            IOU_ANOMALY_THRESH = 0.5

            for _, row in frame_objs.iterrows():
                scores = {k: row[k] if k in row else None for k in ['score_if', 'score_gmm', 'score_cadi', 'score_avg']}
                iou = compute_mask_iou(row['bbox'], frame_mask)

                is_mask_anomaly = int(iou >= IOU_ANOMALY_THRESH)

                # Extract original row
                row_data = row.to_dict()

                # Ensure all score columns are present (fallback to None if missing)
                scores = {k: row.get(k, None) for k in score_cols}

                # Add computed fields
                row_data.update({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "mask_anomaly": int(is_mask_anomaly),
                    "iou": iou,
                    **scores  # Overwrite or ensure presence of score columns
                })

                results.append(row_data)

                # results.append({
                #     "video_id": video_id,
                #     "frame_idx": frame_idx,
                #     "track_id": row.get("track_id", None),
                #     "bbox": row['bbox'],
                #     "mask_anomaly": int(is_mask_anomaly),
                #     "iou": iou,
                #     **scores
                # })

        if results:
            result_df = pd.DataFrame(results)
            all_results.append(result_df)
            output_csv = os.path.join(OUTPUT_DIR, f"{video_id}_iou_comparison.csv")
            result_df.to_csv(output_csv, index=False)

            video_best_metrics = []
            if len(np.unique(result_df['mask_anomaly'])) < 2:
                print(f"Skipping metrics for {video_id} — only one class in mask_anomaly.")
            else:
                for col in score_cols:
                    if col in result_df.columns:
                        try:
                            auc = roc_auc_score(result_df['mask_anomaly'], result_df[col])
                            ap = average_precision_score(result_df['mask_anomaly'], result_df[col])
                            video_best_metrics.append((col, auc, ap))
                        except Exception as e:
                            print(f"Error evaluating {col} on {video_id}: {e}")

            if video_best_metrics:
                best_auc_method = max(video_best_metrics, key=lambda x: x[1])[0]
                best_ap_method = max(video_best_metrics, key=lambda x: x[2])[0]
                best_method_log.append({
                    'video_id': video_id,
                    'best_auc_method': method_names[best_auc_method],
                    'best_ap_method': method_names[best_ap_method]
                })
                    # print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Error processing {score_file}: {e}")



if all_results:
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = normalize_per_video(all_results_df, score_cols)

    metrics_df = evaluate_soft_scores(all_results_df, score_cols)
    
    metrics_csv = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics to: {metrics_csv}")

    # Correlation analysis
    print("\nScore Correlations:")
    corr_matrix = all_results_df[score_cols].corr()
    print(corr_matrix)

    # Class distribution analysis
    print("\nClass Distribution:")
    print(f"Anomalous frames: {all_results_df['mask_anomaly'].sum():,} "f"({all_results_df['mask_anomaly'].mean():.2%})")
    print(f"Normal frames: {len(all_results_df) - all_results_df['mask_anomaly'].sum():,}")

    if best_method_log:
        best_method_df = pd.DataFrame(best_method_log)
        best_method_csv = os.path.join(OUTPUT_DIR, "best_method_per_video.csv")
        best_method_df.to_csv(best_method_csv, index=False)
        print(f"\nSaved best method log to: {best_method_csv}")
    
    # plot_roc_pr_curves(all_results_df, score_cols, OUTPUT_DIR)
    plot_auc_ap_bar(metrics_df, OUTPUT_DIR)
    plot_score_correlation(all_results_df, score_cols, OUTPUT_DIR)
    plot_roc_pr_curves_overlay(all_results_df, score_cols, OUTPUT_DIR)

