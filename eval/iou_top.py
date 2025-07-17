import pandas as pd
import matplotlib.pyplot as plt

# Load your combined object-level scores file (with 'iou', 'score_*', etc.)
df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv")

# Only consider ground-truth anomalous frames
df = df[df["mask_anomaly"] == 1].copy()

# Models to evaluate
methods = {
    "score_if": "IF",
    "score_gmm": "GMM",
    "score_cadi": "CADI",
    "ensemble_score": "ESEMBLE"
}

plt.figure(figsize=(10, 6))

for col, label in methods.items():
    if col not in df.columns:
        continue

    # Rank detections by model score (descending) and get corresponding IoUs
    ranked_iou = df.sort_values(by=col, ascending=False)["iou"].reset_index(drop=True)
    
    # Optionally limit to top-N
    N = 25
    top_n_iou = ranked_iou.iloc[:N]
    
    plt.plot(range(1, len(top_n_iou) + 1), top_n_iou, label=label)

plt.xlabel("Rank of Detection (Top-N)")
plt.ylabel("IoU with Ground Truth (RBDC)")
plt.title("IoU vs. Rank for Top-N Anomalous Detections")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("iou_vs_rank_topn.png")
plt.show()
