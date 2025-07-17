import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to CSV (adjust if needed)
csv_path = "/home/jlin1/OutlierDetection/testing/frames/output_only_logits/comparison_results/best_method_per_video.csv"
output_dir = os.path.dirname("/home/jlin1/OutlierDetection/output")
histogram_path = os.path.join(output_dir, "best_method_histogram.png")

# Load CSV
df = pd.read_csv(csv_path)

plt.figure(figsize=(8, 5))
df['best_auc_method'].value_counts().plot(kind='bar', edgecolor='black')
plt.title("Best AUC Method per Video")
plt.xlabel("Method")
plt.ylabel("Video Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(histogram_path)
print(f"Saved AUC histogram to: {histogram_path}")

