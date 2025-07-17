# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your scored frame or object-level file
# df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv")

# # Define which score columns to plot
# score_cols = {
#     'score_if': 'IF',
#     'score_gmm': 'GMM',
#     'score_cadi': 'CADI',
#     "ensemble_score": "ESEMBLE"
# }

# # Initialize plot
# plt.figure(figsize=(10, 6))

# # Overlay KDE plots
# for col, label in score_cols.items():
#     if col in df.columns:
#         sns.kdeplot(df[col], label=label, fill=False, linewidth=2)

# # Plot settings
# plt.title("Overlayed KDE of Anomaly Score Distributions")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("kde_score_distributions.png")
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your scored frame or object-level file
# df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv")

# # Define which score columns to plot
# score_cols = {
#     'score_if_qnorm': 'IF',
#     'score_gmm_qnorm': 'GMM',
#     'score_cadi_qnorm': 'CADI',
#     "ensemble_score_qnorm": "ENSEMBLE"
# }

# # Loop over each score column and plot separately
# for col, label in score_cols.items():
#     if col in df.columns:
#         plt.figure(figsize=(8, 5))
#         sns.kdeplot(df[col], fill=True, linewidth=2)
        
#         plt.title(f"KDE of Anomaly Score: {label}")
#         plt.xlabel("Anomaly Score")
#         plt.ylabel("Density")
#         plt.grid(True, linestyle='--', alpha=0.5)
        
#         plt.tight_layout()
#         # Save each plot
#         plt.savefig(f"kde_{label.lower()}_score_distribution.png")
#         plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your scored frame or object-level file
df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv")

# # Define which score columns to plot
# score_cols = {
#     'score_if_qnorm': 'IF',
#     'score_gmm_qnorm': 'GMM',
#     'score_cadi_qnorm': 'CADI',
#     "ensemble_score": "ENSEMBLE"
# }

# # Loop over each score column and plot separately
# for col, label in score_cols.items():
#     if col not in df.columns:
#         print(f"[SKIP] {col} not found in DataFrame")
#         continue

#     plt.figure(figsize=(8, 5))

#     # KDE for normal
#     sns.kdeplot(
#         data=df[df["mask_anomaly"] == 0],
#         x=col,
#         label="Normal",
#         color="blue",
#         linestyle="-",
#         linewidth=2
#     )

#     # KDE for anomalous
#     sns.kdeplot(
#         data=df[df["mask_anomaly"] == 1],
#         x=col,
#         label="Anomalous",
#         color="orange",
#         linestyle="--",
#         linewidth=2
#     )

#     plt.title(f"KDE of Anomaly Scores: {label}")
#     plt.xlabel("Anomaly Score")
#     plt.ylabel("Density")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(title="Label")
#     plt.tight_layout()

#     # Save each plot
#     plt.savefig(f"kde_{label.lower()}_score_distribution_with_labels.png")
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your scored frame or object-level file
df = pd.read_csv(
    "/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv"
)

# Define which score columns to plot
score_cols = {
    'score_if_qnorm': 'IF',
    'score_gmm_qnorm': 'GMM',
    'score_cadi_qnorm': 'CADI',
    # "ensemble_score": "ENSEMBLE"
}

# Initialize figure
plt.figure(figsize=(10, 6))

# Create consistent colors for models
palette = sns.color_palette("tab10", n_colors=len(score_cols))
model_colors = dict(zip(score_cols, palette))

# Plot each model's Normal and Anomalous KDEs
for col, label in score_cols.items():
    if col not in df.columns:
        print(f"[SKIP] {col} not found in DataFrame")
        continue
        

    # Normal
    sns.kdeplot(
        data=df[df["mask_anomaly"] == 0],
        x=col,
        color=model_colors[col],
        linestyle="-",
        linewidth=2,
        label=f"{label} (Normal)",
        clip=(0, 1)
    )

    # Anomalous
    sns.kdeplot(
        data=df[df["mask_anomaly"] == 1],
        x=col,
        color=model_colors[col],
        linestyle="--",
        linewidth=2,
        label=f"{label} (Anomalous)",
        clip=(0, 1)
    )

# Titles and labels
plt.title("Anomaly Score Distributions per Model\nNormal: Solid, Anomalous: Dashed")
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
# Get all line handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Create your custom labels
custom_labels = [
    "IF (Normal)", "IF (Anomalous)",
    "GMM (Normal)", "GMM (Anomalous)",
    "CADI (Normal)", "CADI (Anomalous)",
    "ENSEMBLE (Normal)", "ENSEMBLE (Anomalous)"
]

plt.legend(handles, custom_labels)
# plt.ylim(0, 5)
plt.tight_layout()

# Save figure
plt.savefig("kde_all_models_normal_vs_anomalous.png")
plt.show()

