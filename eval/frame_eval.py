import os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

score_cols = ['score_if', 'score_gmm', 'score_cadi', 'ensemble_score']
method_names = {'score_if': 'IF', 'score_gmm': 'GMM', 'score_cadi': 'CADI', 'ensemble_score': 'FUSION'}
total_frames = 0
total_anomalous_frames = 0

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
# MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
# OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results_frame")
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
# SCORE_DIR = "/home/jlin1/OutlierDetection/UCSDped2/frames/output_only_logits"
SCORE_FILE = "/home/jlin1/OutlierDetection/outputs/shanghai_test/all_scored.csv"
MASKS_DIR = "/home/jlin1/OutlierDetection/shanghaitech(og)/testing/test_pixel_mask"
# MASKS_DIR = "/home/jlin1/OutlierDetection/UCSDped2/mask"
OUTPUT_DIR = os.path.join("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.5  # Default threshold for binarizing scores

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

# Plot summary bar chart
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

def evaluate_frame_level(score_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for score_file in os.listdir(score_dir):
        if not score_file.endswith("_scored.csv"):
            continue
            
        video_id = score_file.replace("_scored.csv", "")
        csv_path = os.path.join(score_dir, score_file)
        npy_path = os.path.join(masks_dir, f"{video_id}.npy")
        
        if not os.path.exists(npy_path):
            print(f"Mask file not found for {video_id}")
            continue

        try:
            # Load data
            df = pd.read_csv(csv_path)
            mask_array = np.load(npy_path)
            
            # Frame-level ground truth (1 if any anomaly in frame)
            frame_gt = np.any(mask_array, axis=(1,2)).astype(int)
            
            # Initialize frame predictions
            frame_preds = {col: np.zeros(len(frame_gt)) for col in score_cols}
            
            # Process each frame
            for frame_idx in range(len(frame_gt)):
                frame_scores = df[df['frame_idx'] == frame_idx]
                
                # If no detections, frame is normal (0)
                if len(frame_scores) == 0:
                    continue
                    
                # For each method, frame is anomalous if ANY detection is anomalous
                for col in score_cols:
                    if col not in frame_scores.columns:
                        continue
                        
                    # Using threshold to binarize scores
                    any_anomalous = (frame_scores[col] > THRESHOLD).any()
                    frame_preds[col][frame_idx] = any_anomalous
            
            # Store results for this video
            results = {
                'video_id': video_id,
                'frame_gt': frame_gt,
                **{f'pred_{col}': frame_preds[col] for col in score_cols}
            }
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
    
    return all_results

def evaluate_framelevel_file(score_file, masks_dir, threshold=0.5):
    all_results = []
    df_all = pd.read_csv(score_file)
    video_ids = df_all['frame_dir'].unique()

    for video_id in video_ids:
        df = df_all[df_all['frame_dir'] == video_id]
        npy_path = os.path.join(masks_dir, f"{video_id}.npy")

        if not os.path.exists(npy_path):
            print(f"Mask file not found for {video_id}")
            continue

        try:
            mask_array = np.load(npy_path)
            frame_gt = np.any(mask_array, axis=(1, 2)).astype(int)
            frame_preds = {col: np.zeros(len(frame_gt)) for col in score_cols}

            for frame_idx in range(len(frame_gt)):
                frame_scores = df[df['frame_idx'] == frame_idx]
                if len(frame_scores) == 0:
                    continue
                for col in score_cols:
                    if col in frame_scores.columns:
                        frame_preds[col][frame_idx] = (frame_scores[col] > threshold).any()

            results = {
                'video_id': video_id,
                'frame_gt': frame_gt,
                **{f'pred_{col}': frame_preds[col] for col in score_cols}
            }
            all_results.append(results)

        except Exception as e:
            print(f"[ERROR] {video_id}: {e}")

    return all_results

def evaluate_framelevel_folder(score_dir, masks_dir, threshold=0.5):
    all_results = []

    for fname in os.listdir(score_dir):
        if not fname.endswith("_scored.csv"):
            continue

        score_file = os.path.join(score_dir, fname)
        print(f"Processing: {score_file}")
        results = evaluate_framelevel_file(score_file, masks_dir, threshold)
        all_results.extend(results)

    return all_results

def compute_metrics(all_results, score_source):
    all_gt = np.concatenate([r['frame_gt'] for r in all_results])
    metrics = []

    is_file = os.path.isfile(score_source)

    for col in score_cols:
        pred_key = f'pred_{col}'
        all_pred = np.concatenate([r[pred_key] for r in all_results])

        all_scores = []

        for r in all_results:
            video_id = r['video_id']
            if is_file:
                score_df = pd.read_csv(score_source)
                score_df = score_df[score_df['frame_dir'] == video_id]
            else:
                score_path = os.path.join(score_source, f"{video_id}_scored.csv")
                if not os.path.exists(score_path):
                    print(f"[WARN] Missing score file for video: {video_id}")
                    continue
                score_df = pd.read_csv(score_path)

            video_df = pd.DataFrame({
                'frame_idx': range(len(r['frame_gt'])),
                'gt': r['frame_gt']
            })

            if col not in score_df.columns:
                print(f"[WARN] Score column {col} not found in data for {video_id}")
                continue

            max_scores = score_df.groupby('frame_idx')[col].max()
            video_df['score'] = video_df['frame_idx'].map(max_scores).fillna(0)
            all_scores.append(video_df)

        if not all_scores:
            print(f"[WARN] No scores found for method {col}")
            continue

        all_score_df = pd.concat(all_scores)
        auc = roc_auc_score(all_score_df['gt'], all_score_df['score'])
        ap = average_precision_score(all_score_df['gt'], all_score_df['score'])

        metrics.append({
            'Method': method_names[col],
            'AUC': auc,
            'AP': ap,
        })

    return pd.DataFrame(metrics)

# def compute_metrics(all_results):
#     # Concatenate all videos
#     all_gt = np.concatenate([r['frame_gt'] for r in all_results])
#     metrics = []
    
#     for col in score_cols:
#         pred_key = f'pred_{col}'
#         all_pred = np.concatenate([r[pred_key] for r in all_results])
        
#         # Binary metrics
#         # precision = precision_score(all_gt, all_pred, zero_division=0)
#         # recall = recall_score(all_gt, all_pred)
#         # f1 = f1_score(all_gt, all_pred)
        
#         # Score-based metrics (using maximum score in frame)
#         all_scores = []
#         for r in all_results:
#             video_df = pd.DataFrame({
#                 'frame_idx': range(len(r['frame_gt'])),
#                 'gt': r['frame_gt']
#             })
#             score_df = pd.read_csv(os.path.join(SCORE_DIR, f"{r['video_id']}_scored.csv"))
            
#             # Get max score per frame
#             max_scores = score_df.groupby('frame_idx')[col].max()
#             video_df['score'] = video_df['frame_idx'].map(max_scores).fillna(0)
#             all_scores.append(video_df)
            
#         score_df = pd.concat(all_scores)
#         auc = roc_auc_score(score_df['gt'], score_df['score'])
#         ap = average_precision_score(score_df['gt'], score_df['score'])
        
#         metrics.append({
#             'Method': method_names[col],
#             'AUC': auc,
#             'AP': ap,
#             # 'Precision': precision,
#             # 'Recall': recall,
#             # 'F1': f1
#         })
    
#     return pd.DataFrame(metrics)

# def construct_score_df(all_results, score_source):
#     score_records = []
#     for r in all_results:
#         video_id = r['video_id']
#         frame_gt = r['frame_gt']
#         score_file = os.path.join(SCORE_DIR, f"{video_id}_scored.csv")
#         score_df = pd.read_csv(score_file)

#         for frame_idx in range(len(frame_gt)):
#             frame_data = {'video_id': video_id, 'frame_idx': frame_idx, 'mask_anomaly': frame_gt[frame_idx]}
#             for col in score_cols:
#                 if col in score_df.columns:
#                     max_score = score_df[score_df['frame_idx'] == frame_idx][col].max()
#                     frame_data[col] = max_score if not np.isnan(max_score) else 0
#                 else:
#                     frame_data[col] = 0
#             score_records.append(frame_data)

#     return pd.DataFrame(score_records)

def construct_score_df(all_results, score_source):
    score_records = []
    is_file = os.path.isfile(score_source)

    if is_file:
        df_all = pd.read_csv(score_source)
        if 'frame_dir' not in df_all.columns:
            raise ValueError("Combined score file must contain a 'video_id' column.")
    else:
        df_all = None  # Only used if loading individual files

    for r in all_results:
        video_id = r['video_id']
        frame_gt = r['frame_gt']

        if is_file:
            score_df = df_all[df_all['frame_dir'] == video_id]
        else:
            score_file = os.path.join(score_source, f"{video_id}_scored.csv")
            if not os.path.exists(score_file):
                print(f"[WARN] Score file not found for {video_id}")
                continue
            score_df = pd.read_csv(score_file)

        for frame_idx in range(len(frame_gt)):
            frame_data = {'video_id': video_id, 'frame_idx': frame_idx, 'mask_anomaly': frame_gt[frame_idx]}
            for col in score_cols:
                if col in score_df.columns:
                    max_score = score_df[score_df['frame_idx'] == frame_idx][col].max()
                    frame_data[col] = max_score if not np.isnan(max_score) else 0
                else:
                    frame_data[col] = 0
            score_records.append(frame_data)

    
    print("=== Score DF Columns ===")
    print(score_df.columns)
    print("=== Unique Scores ===")
    for col in score_cols:
        if col in score_df.columns:
            print(f"{col} unique values:", score_df[col].unique()[:5])

    return pd.DataFrame(score_records)

# Run evaluation

# all_results = evaluate_frame_level(SCORE_DIR, MASKS_DIR, OUTPUT_DIR)
all_results = evaluate_framelevel_file(SCORE_FILE, MASKS_DIR)

metrics_df = compute_metrics(all_results, SCORE_FILE)
all_results_df = construct_score_df(all_results, SCORE_FILE)

# Save and display results
print("\nFrame-Level Evaluation Metrics:")
print(metrics_df.round(4))

corr_matrix = all_results_df[score_cols].corr()
print(corr_matrix)

metrics_csv = os.path.join(OUTPUT_DIR, "frame_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nSaved metrics to: {metrics_csv}")

plot_score_correlation(all_results_df, score_cols, OUTPUT_DIR)
plot_roc_pr_curves_overlay(all_results_df, score_cols, OUTPUT_DIR)
