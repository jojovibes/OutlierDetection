import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your frame-level scores file
df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results.csv")

# Models to plot
methods = {
    "score_if": "IF",
    "score_gmm": "GMM",
    "score_cadi": "CADI",
    "ensemble_score": "ENSEMBLE"
}

# Limit to top N
N = 160576

plt.figure(figsize=(12, 6))

for col, label in methods.items():
    if col not in df.columns:
        continue
    
    sorted_scores = np.sort(df[col].values)[::-1]  # descending
    top_scores = sorted_scores[:N]  # only take top N
    
    plt.plot(top_scores, label=label, linewidth=2)

# Mark Top-K (if you want)
K_values = [22705]
for K in K_values:
    plt.axvline(x=K, color='gray', linestyle='--', alpha=0.7)
    plt.text(K + 1, plt.ylim()[1]*0.95, f"K={K}", rotation=90, color='gray')

plt.xlabel(f"Top {N} Frames Ranked by Anomaly Score (Descending)")
plt.ylabel("Anomaly Score")
plt.title(f"Top-{N} Frame-Level Anomaly Scores per Model")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("topk_all_score_ranking_top20000.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load
df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results.csv")

# Define models and labels
models = {
    "score_gmm": "GMM",
    "ensemble_score": "FUSION",
    "score_if": "IF",
    "score_cadi": "CADI"
}

# Check label column
if "mask_anomaly" not in df.columns:
    raise ValueError("CSV must have 'mask_anomaly' column with 0/1 labels.")

# Limit to top N
N = 160576
K_values = [22705]

# Loop through each model
for score_col, model_label in models.items():
    if score_col not in df.columns:
        print(f"Skipping {model_label}: column '{score_col}' not found.")
        continue

    scores = df[score_col].values
    y_true = df["mask_anomaly"].values

    # Sort descending
    sorted_indices = np.argsort(-scores)
    sorted_indices = sorted_indices[:N]
    sorted_scores = scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    anomaly_indices = [i for i, lbl in enumerate(sorted_labels) if lbl == 1]
    normal_indices = [i for i, lbl in enumerate(sorted_labels) if lbl == 0]

    plt.figure(figsize=(12, 6))

    # Plot normal points (blue, faint)
    plt.scatter(
        normal_indices,
        [sorted_scores[i] for i in normal_indices],
        c="blue",
        s=10,
        alpha=0.3,
        label="Normal"
    )

    # Plot anomaly points (red, big, opaque, with black edge)
    plt.scatter(
        anomaly_indices,
        [sorted_scores[i] for i in anomaly_indices],
        c="red",
        s=10,
        alpha=0.3,
        # edgecolor="black",
        linewidth=0.5,
        label="Anomaly"
    )


    # colors = ["red" if lbl == 1 else "blue" for lbl in sorted_labels]

    # plt.figure(figsize=(12, 6))
    # plt.scatter(
    #     range(len(sorted_scores)),
    #     sorted_scores,
    #     c=colors,
    #     s=10,
    #     alpha=0.8,
    #     label="Red=Anomaly, Blue=Normal"
    # )
    plt.plot(sorted_scores, color="black", alpha=0.3, linewidth=1)

    # Mark Top-K thresholds
    for K in K_values:
        plt.axvline(x=K, color='gray', linestyle='--', alpha=0.7)
        plt.text(
            K + 1,
            plt.ylim()[1]*0.95,
            f"K={K}",
            rotation=90,
            color='gray'
        )

    plt.xlabel(f"Top {N} Frames Ranked by Anomaly Score (Descending)")
    plt.ylabel("Anomaly Score")
    plt.title(f"Top-{N} Frame-Level Anomaly Scores with Labels ({model_label})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    filename = f"topk_score_ranking_{model_label.lower()}_top{N}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")
