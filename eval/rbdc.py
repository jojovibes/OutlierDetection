import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming this is your scored and matched dataframe
# Columns: 'mask_anomaly', 'iou', 'score_if', 'score_gmm', ...

df = pd.read_csv("/home/jlin1/OutlierDetection/outputs/shanghai_test/comparison_results_frame/detection_results_and_features.csv")

# Only consider detections where the GT mask is anomalous
anomalies_df = df[df["mask_anomaly"] == 1].copy()

# Threshold each method's score to select top anomalous detections (e.g., top 5%)
percentile = 85
rbdc_data = []

for score_col, label in {
    "score_if_qnorm": "IF",
    "score_gmm_qnorm": "GMM",
    "score_cadi_qnorm": "CADI",
    "ensemble_score": "ESEMBLE"
}.items():
    if score_col not in anomalies_df.columns:
        continue
    
    score_thresh = anomalies_df[score_col].quantile(percentile / 100.0)
    selected = anomalies_df[anomalies_df[score_col] >= score_thresh]
    
    for iou in selected["iou"]:
        rbdc_data.append({"Model": label, "RBDC": iou})

rbdc_df = pd.DataFrame(rbdc_data)
rbdc_df.to_csv("spatial_robustness_rbdc.csv", index=False)
print(rbdc_df)

# Group by model and compute mean and standard deviation of RBDC
rbdc_summary = rbdc_df.groupby("Model")["RBDC"].agg(['mean', 'std']).reset_index()
rbdc_summary.columns = ["Model", "Mean_RBDC", "SD_RBDC"]

# Print or save
print(rbdc_summary)
rbdc_summary.to_csv("rbdc_mean_sd_per_model.csv", index=False)



# --- Boxplot ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="Model", y="RBDC", hue="Model", data=rbdc_df, palette="pastel", legend=False)
plt.title("Spatial Robustness Comparison (RBDC for Top Anomalous Detections)")
plt.ylabel("IoU with Ground Truth (RBDC)")
plt.xlabel("Anomaly Detection Method")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("spatial_robustness_boxplot.png")
plt.show()
