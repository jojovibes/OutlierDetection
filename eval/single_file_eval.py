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

# score_cols = ['score_if_norm', 'score_gmm_norm', 'score_cadi_norm', 'ensemble_score']
# method_names = {'score_if_norm': 'IF', 'score_gmm_norm': 'GMM', 'score_cadi_norm': 'CADI', 'ensemble_score': 'FUSION'}

score_cols = ['score_if_qnorm', 'score_gmm_qnorm', 'score_cadi_qnorm', 'ensemble_score_qnorm']
method_names = {'score_if_qnorm': 'IF', 'score_gmm_qnorm': 'GMM', 'score_cadi_qnorm': 'CADI', 'ensemble_score_qnorm': 'ENSEMBLE'}

total_frames = 0
total_anomalous_frames = 0

SCORE_FILE = "/home/jlin1/OutlierDetection/outputs/shanghai_test/alpha_0.9_all_scored.csv"
MASKS_DIR = "/home/jlin1/OutlierDetection/shanghaitech(og)/testing/test_pixel_mask"

OUTPUT_DIR = os.path.join("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.5  

from scipy.ndimage import label

def split_mask_into_blobs(mask):
    binary_mask = mask > 0
    labeled_mask, num_blobs = label(binary_mask)
    return [(labeled_mask == i).astype(np.uint8) for i in range(1, num_blobs + 1)]

def plot_roc_pr_curves_overlay(df, score_cols, output_dir, name):
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
    plt.savefig(os.path.join(output_dir, f"overlayed_roc_curve_{name}.png"))
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
    plt.savefig(os.path.join(output_dir, f"overlayed_pr_curve_{name}.png"))
    plt.close()

def plot_auc_ap_bar(metrics_df, output_dir, name):
    plt.figure(figsize=(8, 5))
    df_melted = metrics_df.melt(id_vars="Method", value_vars=["AUC", "AP"], var_name="Metric", value_name="Score")
    sns.barplot(x="Method", y="Score", hue="Metric", data=df_melted)
    plt.title("AUC and Average Precision per Method")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"auc_ap_barplot_{name}.png"))
    plt.close()

def plot_score_correlation(df, score_cols, output_dir, name):
    corr = df[score_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={'label': 'Correlation'})
    plt.title("Score Correlation Between Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"score_correlation_heatmap_{name}.png"))
    plt.close()

def compute_mask_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    mask_bool = mask > 0
    intersection = np.logical_and(mask_bool, bbox_mask).sum()

    union = (x2 - x1)*(y2 - y1)

    return intersection / union if union > 0 else 0.0

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

def compute_metrics(all_results, score_df):
    all_gt = np.concatenate([r['frame_gt'] for r in all_results])
    metrics = []

    for col in score_cols:
        all_scores = []

        for r in all_results:
            video_id = r['video_id']
            video_df = pd.DataFrame({
                'frame_idx': range(len(r['frame_gt'])),
                'gt': r['frame_gt']
            })

            score_subset = score_df[score_df['frame_dir'] == video_id]
            max_scores = score_subset.groupby('frame_idx')[col].max()
            video_df[col] = video_df['frame_idx'].map(max_scores).fillna(0)

            all_scores.append(video_df[['frame_idx', 'gt', col]])

        combined = pd.concat(all_scores)
        auc = roc_auc_score(combined['gt'], combined[col])
        ap = average_precision_score(combined['gt'], combined[col])

        metrics.append({
            'Method': method_names[col],
            'AUC': auc,
            'AP': ap
        })

    return pd.DataFrame(metrics)

def construct_score_df(all_results, score_df):
    score_records = []

    for r in all_results:
        video_id = r['video_id']
        frame_gt = r['frame_gt']
        subset = score_df[score_df['frame_dir'] == video_id]

        for frame_idx in range(len(frame_gt)):
            frame_data = {'video_id': video_id, 'frame_idx': frame_idx, 'mask_anomaly': frame_gt[frame_idx]}
            frame_subset = subset[subset['frame_idx'] == frame_idx]

            for col in score_cols:
                if col in frame_subset.columns:
                    max_score = frame_subset[col].max()
                    frame_data[col] = max_score if not np.isnan(max_score) else 0
                else:
                    frame_data[col] = 0

            score_records.append(frame_data)

    return pd.DataFrame(score_records)

def compute_detection_level_metrics(score_df, mask_dir, iou_thresh=0.5):
    
    print("\nDetection-Level Evaluation!!!:")

    results = []

    for video_id in score_df["frame_dir"].unique():
        df_video = score_df[score_df["frame_dir"] == video_id].copy()
        npy_path = os.path.join(MASKS_DIR, f"{video_id}.npy")

        if not os.path.exists(npy_path):
            print(f"[WARN] Missing mask file: {npy_path}")
            continue

        mask_array = np.load(npy_path)

        for _, row in df_video.iterrows():
            frame_idx = int(row["frame_idx"])

            if frame_idx >= len(mask_array):
                print(f"[WARN] Frame {frame_idx} out of range in {video_id}")
                continue

            mask = mask_array[frame_idx]
            bbox = row["bbox"] if isinstance(row["bbox"], list) else ast.literal_eval(row["bbox"])
            # iou = compute_mask_iou(bbox,mask)

            blobs = split_mask_into_blobs(mask)

            # Match detection with highest-IoU blob
            ious = [compute_mask_iou(bbox, blob) for blob in blobs]
            iou = max(ious) if ious else 0.0
            
            if iou >= iou_thresh:
                mask_anomaly = 1 #int(mask.any()) 
            else:
                mask_anomaly = 0

            # if iou > 0:
            #     print(f"mask_anomaly = {mask_anomaly} ground_truth = {int(mask.any())} iou = {iou}")

            record = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "bbox": bbox,
                "iou": iou,
                "mask_anomaly": mask_anomaly,
            }

            for col in score_cols:
                record[col] = row.get(col, 0)

            results.append(record)


    det_df = pd.DataFrame(results)
    det_df.to_csv(os.path.join(OUTPUT_DIR, "video_mask_iou_eval.csv"), index=False)

    metrics_df = evaluate_soft_scores(det_df, score_cols)

    return metrics_df, det_df

def evaluate_soft_scores(df, score_cols):
    results = []

    for col in score_cols:
        if col not in df.columns:
            continue

        # Force numeric, drop rows with missing or invalid data
        df[col] = pd.to_numeric(df[col], errors='coerce')
        valid = df[['mask_anomaly', col]].dropna()

        if valid[col].nunique() < 2:
            print(f"[SKIP] {col}: not enough variation in scores")
            continue

        try:
            auc = roc_auc_score(valid["mask_anomaly"], valid[col])
            ap = average_precision_score(valid["mask_anomaly"], valid[col])
            results.append({"Method": method_names[col], "AUC": auc, "AP": ap})
        except Exception as e:
            print(f"[ERROR] {col}: {e}")
            continue

    if not results:
        # Return empty DataFrame with expected columns to avoid KeyError in melt
        return pd.DataFrame(columns=["Method", "AUC", "AP"])

    return pd.DataFrame(results)
   
def construct_score_df_multiagg(all_results, score_df):
    score_records = []

    for r in all_results:
        video_id = r['video_id']
        frame_gt = r['frame_gt']
        subset = score_df[score_df['frame_dir'] == video_id]

        for frame_idx in range(len(frame_gt)):
            frame_data = {'video_id': video_id, 'frame_idx': frame_idx, 'mask_anomaly': frame_gt[frame_idx]}
            frame_subset = subset[subset['frame_idx'] == frame_idx]

            for col in score_cols:
                if col in frame_subset.columns:
                    scores = frame_subset[col].dropna()
                    frame_data[f"{col}_max"] = scores.max() if not scores.empty else 0
                    frame_data[f"{col}_mean"] = scores.mean() if not scores.empty else 0
                    frame_data[f"{col}_count"] = (scores > THRESHOLD).sum()
                else:
                    frame_data[f"{col}_max"] = 0
                    frame_data[f"{col}_mean"] = 0
                    frame_data[f"{col}_count"] = 0

            score_records.append(frame_data)

    return pd.DataFrame(score_records)

def plot_robustness_curve(df, col_base, y_true_col, output_dir, name):
    plt.figure(figsize=(8, 6))
    thresholds = np.linspace(0, 1, 50)

    for col in col_base:
        if col not in df.columns:
            continue

        tprs = []
        for t in thresholds:
            preds = df[col] > t
            tpr = recall_score(df[y_true_col], preds, zero_division=0)
            tprs.append(tpr)

        plt.plot(thresholds, tprs, label=method_names.get(col.replace('_max','').replace('_mean','').replace('_count',''), col))

    plt.xlabel("Threshold")
    plt.ylabel("TPR (Recall)")
    plt.title("Robustness Curve (TPR vs Threshold)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"robustness_curve_{name}.png"))
    plt.close()

def plot_temporal_scores_overlay(df, score_cols, output_dir, name="temporal_overlay", max_videos=10):
 
    # Match full column names to display names
    # method_names = {
    #     'score_if_max': 'IF',
    #     'score_gmm_max': 'GMM',
    #     'score_cadi_max': 'CADI',
    #     'ensemble_score_max': 'ENSEMBLE'
    # }

    method_names = {
    'score_if_qnorm_max': 'IF',
    'score_gmm_qnorm_max': 'GMM',
    'score_cadi_qnorm_max': 'CADI',
    'ensemble_score_qnorm_max': 'ENSEMBLE'
    }


    # Select top videos with most anomalies
    top_anomalous = (
        df.groupby('video_id')['mask_anomaly']
        .sum()
        .sort_values(ascending=False)
        .head(max_videos)
        .index
    )

    for vid in top_anomalous:
        subset = df[df['video_id'] == vid].sort_values("frame_idx")

        plt.figure(figsize=(15, 5))

        # Plot each model's score
        for col in score_cols:
            if col not in subset.columns:
                continue
            label = method_names.get(col, col)
            plt.plot(subset['frame_idx'], subset[col], label=label)

        # Highlight anomaly regions
        plt.fill_between(subset['frame_idx'], 0, 1, where=subset['mask_anomaly'] == 1, alpha=0.2, color='red', label='Anomaly (GT)')

        handles, labels = plt.gca().get_legend_handles_labels()

        # Create your custom labels
        custom_labels = [
            "IF", "GMM", "CADI", "ESEMBLE"
        ]

        plt.legend(handles, custom_labels,loc="lower left")

        plt.title(f"Temporal Score Overlay - {vid}")
        plt.xlabel("Frame Index")
        plt.ylabel("Score")
        plt.ylim(0, 1)
    
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_{vid}.png"))
        plt.close()

# def plot_temporal_scores_overlay(df,score_cols,output_dir,name="temporal_overlay",max_videos=5):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import os

#     # Map model names to display names
#     method_names = {
#         'score_if_max': 'IF',
#         'score_gmm_max': 'GMM',
#         'score_cadi_max': 'CADI',
#         'ensemble_score_max': 'ENSEMBLE'
#     }

#     # Select top videos with most anomalies
#     top_anomalous_videos = (
#         df.groupby('video_id')['mask_anomaly']
#         .sum()
#         .sort_values(ascending=False)
#         .head(max_videos)
#         .index
#     )

#     # Use a consistent color palette
#     palette = sns.color_palette("tab10", n_colors=len(score_cols))
#     model_colors = dict(zip(score_cols, palette))

#     for vid in top_anomalous_videos:
#         subset = df[df['video_id'] == vid].sort_values("frame_idx")

#         plt.figure(figsize=(15, 5))

#         # Plot each model's score
#         for col in score_cols:
#             if col not in subset.columns:
#                 continue
#             label = method_names.get(col, col)
#             plt.plot(
#                 subset['frame_idx'],
#                 subset[col],
#                 label=label,
#                 color=model_colors[col]
#             )

#         # Highlight anomaly regions
#         plt.fill_between(
#             subset['frame_idx'],
#             0, 1,
#             where=subset['mask_anomaly'] == 1,
#             alpha=0.2,
#             color='red',
#             label="Anomaly (GT)"
#         )

#         plt.title(f"Temporal Score Overlay - {vid}")
#         plt.xlabel("Frame Index")
#         plt.ylabel("Normalized Anomaly Score")
#         plt.ylim(0, 1)
#         plt.legend(loc="upper right")
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.tight_layout()

#         # Save
#         filename = os.path.join(output_dir, f"{name}_{vid}.png")
#         plt.savefig(filename)
#         plt.close()

#         print(f"Saved temporal overlay for video {vid} to {filename}")


def plot_score_heatmap_overlay(df, masks_dir, score_col, output_dir,score_cols=None, max_videos=5, max_frames=5,box_color="white"):
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize

    if score_cols is None:
        score_cols = ['score_if_qnorm', 'score_gmm_qnorm', 'score_cadi_qnorm', 'ensemble_score_qnorm']

    # Map base names to labels
    method_names = {
    'score_if_qnorm_max': 'IF',
    'score_gmm_qnorm_max': 'GMM',
    'score_cadi_qnorm_max': 'CADI',
    'ensemble_score_qnorm_max': 'ENSEMBLE'
    }

    anomaly_videos = df.groupby("video_id")["mask_anomaly"].sum()
    selected_videos = anomaly_videos[anomaly_videos > 0].sort_values(ascending=False).head(max_videos).index

    for vid in selected_videos:
        mask_path = os.path.join(masks_dir, f"{vid}.npy")
        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {vid}")
            continue

        mask_array = np.load(mask_path)
        subset = df[df["video_id"] == vid]
        subset = subset[subset["mask_anomaly"] == 1].sort_values(score_col, ascending=False)
        shown_frames = subset["frame_idx"].unique()[:max_frames]

        for idx in shown_frames:
            frame_rows = subset[subset["frame_idx"] == idx]
            if frame_rows.empty:
                continue

            # Use first row for frame-level scores
            row = frame_rows.iloc[0]
            mask = mask_array[idx]
            main_score = row[score_col]

            # Build multi-model score display
            score_strs = []
            for col in score_cols:
                for base in model_labels:
                    if col.startswith(base):
                        label = model_labels[base]
                        score_strs.append(f"{label}: {row.get(col, 0):.2f}")
                        break

            title_str = f"{vid} Frame {idx}\n" + " | ".join(score_strs)

            # Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(mask, cmap='gray', alpha=0.5)
            norm = Normalize(vmin=0, vmax=1)
            ax.imshow(np.ones_like(mask) * main_score, cmap='hot', alpha=0.5, norm=norm)

            # Draw bounding boxes
            for _, det_row in frame_rows.iterrows():
                bbox = det_row.get("bbox", None)
                if isinstance(bbox, str):
                    try:
                        bbox = ast.literal_eval(bbox)
                    except Exception:
                        continue
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor=box_color, facecolor='none'
                )
                ax.add_patch(rect)

            ax.set_title(title_str, fontsize=10)
            ax.axis("off")
            plt.tight_layout()
            filename = f"heatmap_{vid}_{idx}_{score_col}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

def compute_temporal_alignment(df, score_col, threshold=0.5):
    delays = []

    for vid in df['video_id'].unique():
        subset = df[df['video_id'] == vid].sort_values('frame_idx')
        gt = subset['mask_anomaly'].values
        preds = (subset[score_col] > threshold).astype(int)

        if 1 not in gt:
            continue

        first_gt = np.argmax(gt)
        first_pred = np.argmax(preds) if 1 in preds else len(gt)
        delay = first_pred - first_gt
        delays.append(delay)

    return delays

def plot_score_distribution(df, score_cols, output_dir, name="score_distribution"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    for col in score_cols:
        if col not in df.columns:
            print(f"[SKIP] {col} not in DataFrame")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df, x=col, hue="mask_anomaly", bins=50, kde=True,
            palette={0: "skyblue", 1: "orange"},
            stat="density", common_norm=False, alpha=0.7
        )
        plt.title(f"Score Distribution - {col}")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend(title="Anomalous", labels=["Normal", "Anomalous"])
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"score_distribution_{col}_{name}.png")
        plt.savefig(out_path)
        plt.close()

# def plot_combined_score_distribution(df, score_cols, output_dir, name="score_distribution_all"):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import os

#     # Check that all columns exist
#     for col in score_cols:
#         if col not in df.columns:
#             raise ValueError(f"{col} not found in DataFrame columns!")

#     # Melt to long format:
#     df_long = df.melt(
#         id_vars="mask_anomaly",
#         value_vars=score_cols,
#         var_name="Model",
#         value_name="Score"
#     )

#     plt.figure(figsize=(10, 6))
#     sns.histplot(
#         data=df_long,
#         x="Score",
#         hue="Model",
#         bins=50,
#         kde=True,
#         stat="density",
#         common_norm=False,
#         alpha=0.6
#     )

#     plt.title("Score Distribution Across All Models")
#     plt.xlabel("Anomaly Score")
#     plt.ylabel("Density")
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(title="Model")
#     plt.tight_layout()

#     out_path = os.path.join(output_dir, f"score_distribution_combined_{name}.png")
#     plt.savefig(out_path)
#     plt.close()

def plot_combined_score_distribution(df, score_cols, output_path, name="score_distribution_combined"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Check columns
    for col in score_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in DataFrame columns!")

    # Melt DataFrame
    df_long = df.melt(
        id_vars="mask_anomaly",
        value_vars=score_cols,
        var_name="Model",
        value_name="Score"
    )

    # Map label names
    df_long["Label"] = df_long["mask_anomaly"].map({0: "Normal", 1: "Anomalous"})

    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Get palette for models
    palette = sns.color_palette("tab10", n_colors=len(score_cols))
    model_colors = dict(zip(score_cols, palette))

    # Plot each combination separately to customize styles
    for model in score_cols:
        for label in ["Normal", "Anomalous"]:
            subset = df_long[(df_long["Model"] == model) & (df_long["Label"] == label)]
            linestyle = "-" if label == "Normal" else "--"
            sns.kdeplot(
                data=subset,
                x="Score",
                color=model_colors[model],
                linestyle=linestyle,
                linewidth=2,
                label=f"{model} ({label})",
                fill=False,
                clip=(0,1)
            )

    # Titles and labels
    plt.title("Score Distributions per Model (Normal: solid, Anomalous: dashed)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()

    # Save
    out_file = f"{output_path}/{name}.png"
    plt.savefig(out_file)
    plt.close()

    print(f"Plot saved to {out_file}")

def plot_score_distribution_overlay(df, score_cols, output_path, name="score_distribution_all_models"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Check columns
    for col in score_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not in DataFrame columns!")

    # Melt DataFrame to long format
    df_long = df.melt(
        id_vars="mask_anomaly",
        value_vars=score_cols,
        var_name="Model",
        value_name="Score"
    )

    # Map label names
    df_long["Label"] = df_long["mask_anomaly"].map({0: "Normal", 1: "Anomalous"})

    # Define colors for models
    palette = sns.color_palette("tab10", n_colors=len(score_cols))
    model_colors = dict(zip(score_cols, palette))

    # Start figure
    plt.figure(figsize=(10, 6))

    # sns.kdeplot(df[df["mask_anomaly"]==0]["score_cadi_qnorm"], label="Normal")
    # sns.kdeplot(df[df["mask_anomaly"]==1]["score_cadi_qnorm"], label="Anomalous")


    # Plot each combination separately to control linestyle and color
    for model in score_cols:
        for label in ["Normal", "Anomalous"]:
            subset = df_long[
                (df_long["Model"] == model) &
                (df_long["Label"] == label)
            ]
            linestyle = "-" if label == "Normal" else "--"
            sns.kdeplot(
                data=subset,
                x="Score",
                color=model_colors[model],
                linestyle=linestyle,
                linewidth=2,
                label=f"{model} ({label})",
                clip=(0, 1)
            )

    # Configure axes
    plt.xlim(0, 1)
    plt.xlabel("Normalized Anomaly Score (0–1)")
    plt.ylabel("Density")
    plt.title("Score Distributions per Model (Normal=Solid, Anomalous=Dashed)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save
    out_file = f"{output_path}/{name}.png"
    plt.savefig(out_file)
    plt.close()

    print(f"Plot saved to {out_file}")

def plot_histogram_overlay(df, score_cols, output_path, name="score_histograms_overlay"):
    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("tab10", n_colors=len(score_cols))
    model_colors = dict(zip(score_cols, palette))

    for col in score_cols:
        for label in [0, 1]:
            subset = df[df["mask_anomaly"] == label]
            linestyle = "-" if label == 0 else "--"
            sns.kdeplot(
                subset[col],
                color=model_colors[col],
                linestyle=linestyle,
                linewidth=2,
                label=f"{col} ({'Normal' if label==0 else 'Anomalous'})",
                clip=(0,1)
            )

    plt.xlabel("Normalized Anomaly Score")
    plt.ylabel("Density")
    plt.title("Overlayed Score Distributions (Normal: solid, Anomalous: dashed)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()

    out_file = f"{output_path}/{name}.png"
    plt.savefig(out_file)
    plt.close()

    print(f"Saved overlayed histogram to {out_file}")

score_df = pd.read_csv(SCORE_FILE)
score_df["ensemble_score_qnorm"] = score_df["ensemble_score"].rank(method="average", pct=True)

# FRAME-LEVEL EVALUATION
all_results = evaluate_framelevel_file(SCORE_FILE, MASKS_DIR)
metrics_df = compute_metrics(all_results, score_df)
all_results_df = construct_score_df(all_results, score_df)

print(f"Total frames: {len(all_results_df)} | Anomalous frames: {all_results_df['mask_anomaly'].sum()}")

print("\nFrame-Level Evaluation Metrics:")
print(metrics_df.round(4))

corr_matrix = all_results_df[score_cols].corr()
print("\nFrame-Level Score Correlation:")
print(corr_matrix.round(2))

# Save frame-level metrics
metrics_csv = os.path.join(OUTPUT_DIR, "frame_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"Saved frame-level metrics to: {metrics_csv}")

plot_score_correlation(all_results_df, score_cols, OUTPUT_DIR, "frame")
plot_roc_pr_curves_overlay(all_results_df, score_cols, OUTPUT_DIR, "frame")
plot_auc_ap_bar(metrics_df, OUTPUT_DIR, "frame")

score_df_agg = construct_score_df_multiagg(all_results, score_df)



# plot_temporal_scores_overlay(score_df_agg,['score_if_qnorm_max', 'score_gmm_qnorm_max', 'score_cadi_qnorm_max', 'ensemble_score_qnorm_max'], OUTPUT_DIR)

# plot_robustness_curve(score_df_agg, [f"{col}_max" for col in score_cols], "mask_anomaly", OUTPUT_DIR, "frame")
# # plot_temporal_scores(score_df_agg, 'score_gmm_max', OUTPUT_DIR, "frame")
# plot_score_heatmap_overlay(score_df_agg, MASKS_DIR, 'score_gmm_max', OUTPUT_DIR)
# plot_score_heatmap_overlay(score_df_agg, MASKS_DIR, 'score_gmm_max', OUTPUT_DIR,
#                            score_cols=['score_if_qnorm_mean', 'score_gmm_qnorm_mean', 'score_cadi_qnorm_mean', 'ensemble_score_qnorm_mean'])

# print(f"Mean anomaly detection delay: {np.mean(delays):.2f} frames")

# DETECTION-LEVEL EVALUATION
bbox_cols = ['x1', 'y1', 'x2', 'y2']
invalid_bbox_mask = score_df[bbox_cols].isna().any(axis=1)
num_invalid = invalid_bbox_mask.sum()

if num_invalid > 0:
    print(f"Found {num_invalid} detections with invalid bboxes:")
    print(score_df.loc[invalid_bbox_mask, ['frame_dir', 'frame_idx'] + bbox_cols].head())

# Clean and prepare bbox
score_df = score_df[~invalid_bbox_mask].copy()
score_df["bbox"] = score_df.apply(lambda row: [int(row.x1), int(row.y1), int(row.x2), int(row.y2)], axis=1)

# Evaluate
det_metrics_df, det_results_df = compute_detection_level_metrics(score_df, MASKS_DIR)
det_results_df["mask_anomaly"] = det_results_df["mask_anomaly"].astype(int)

# # Thresholds
# HIGH_IOU_THRESHOLD = 0.97   # consider IoU >= 0.5 as good detection
# LOW_IOU_THRESHOLD = 0.6   # consider IoU <= 0.1 as failure

# # Filter: Anomalous frames with high IoU
# high_iou_anomalies = det_results_df[
#     (det_results_df["mask_anomaly"] == 1) &
#     (det_results_df["iou"] >= HIGH_IOU_THRESHOLD)
# ]

# # Filter: Anomalous frames with low IoU
# low_iou_anomalies = det_results_df[
#     (det_results_df["mask_anomaly"] == 1) &
#     (det_results_df["iou"] <= LOW_IOU_THRESHOLD)
# ]

# # Display results
# print("\n=== Anomalous Frames with HIGH IoU (Correct detections) ===")
# if not high_iou_anomalies.empty:
#     for idx, row in high_iou_anomalies.iterrows():
#         print(
#             f"Video: {row['video_id']} | Frame: {int(row['frame_idx'])} | IoU: {row['iou']:.3f}"
#         )
# else:
#     print("None found.")


# print("\n=== Anomalous Frames with LOW IoU (Missed detections) ===")

# if not low_iou_anomalies.empty:
#     for idx, row in low_iou_anomalies.iterrows():
#         print(
#             f"Video: {row['video_id']} | Frame: {int(row['frame_idx'])} | IoU: {row['iou']:.3f}"
#         )
# else:
#     print("None found.")


corr_matrix = det_results_df[score_cols].corr()
print("\nDetection-Level Score Correlation:")
print(corr_matrix.round(2))

print(f"\nTotal detections: {len(det_results_df)} | Anomalous detections: {det_results_df['mask_anomaly'].sum()}")
print("\nDetection-Level Evaluation Metrics:")
print(det_metrics_df.round(4))

# Save detection-level metrics
det_metrics_csv = os.path.join(OUTPUT_DIR, "detection_metrics.csv")
det_metrics_df.to_csv(det_metrics_csv, index=False)
print(f"Saved detection metrics to: {det_metrics_csv}")

# Save raw detection results
det_results_csv = os.path.join(OUTPUT_DIR, "detection_results.csv")
det_results_df.to_csv(det_results_csv, index=False)
print(f"Saved detection results to: {det_results_csv}")

# Save merged results (optional)
det_results_and_feat_df = pd.concat([det_results_df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
det_results_and_feat_csv = os.path.join(OUTPUT_DIR, "detection_results_and_features.csv")
det_results_and_feat_df.to_csv(det_results_and_feat_csv, index=False)
print(f"Saved merged detection + features to: {det_results_and_feat_csv}")

# Visualizations
plot_score_correlation(det_results_df, score_cols, OUTPUT_DIR, "detection")
plot_roc_pr_curves_overlay(det_results_df, score_cols, OUTPUT_DIR, "detection")
plot_auc_ap_bar(det_metrics_df, OUTPUT_DIR, "detection")
plot_score_distribution(det_results_df, score_cols, OUTPUT_DIR, "detection")

plot_temporal_scores_overlay(
    score_df_agg,
    ['score_if_max', 'score_gmm_max', 'score_cadi_max', 'ensemble_score_max'],
    OUTPUT_DIR,
    name="temporal_overlay",
    max_videos=5
)


plot_score_distribution(score_df_agg, ['score_if_qnorm', 'score_gmm_qnorm', 'score_cadi_qnorm', 'ensemble_score_qnorm'], OUTPUT_DIR, "frame")

score_cols = ["score_if_qnorm", "score_gmm_qnorm", "score_cadi_qnorm", "ensemble_score"]
plot_combined_score_distribution(det_results_df, score_cols, OUTPUT_DIR)

score_cols = ["score_if_qnorm", "score_gmm_qnorm", "score_cadi_qnorm", "ensemble_score_qnorm"]
plot_score_distribution_overlay(
    det_results_df,
    score_cols,
    OUTPUT_DIR,
    name="all_models_overlay"
)

plot_histogram_overlay(
    det_results_df,
    ["score_if_qnorm", "score_gmm_qnorm", "score_cadi_qnorm", "ensemble_score_qnorm"],
    output_path=OUTPUT_DIR
)


print(score_df_agg.columns.tolist())
