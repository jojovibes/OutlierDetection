import os
import pandas as pd
import glob

# Path to the folder with *_iou_comparison.csv files
INPUT_DIR = "/home/jlin1/OutlierDetection/testing/frames/output_only_logits/comparison_results_detection"
OUTPUT_FILE = os.path.join(INPUT_DIR, "combined_iou_scores.csv")

# Load all *_iou_comparison.csv files
csv_files = glob.glob(os.path.join(INPUT_DIR, "*_iou_comparison.csv"))

# Combine them
all_dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    
    # Optional: add video_id if not already present
    if "video_id" not in df.columns:
        video_id = os.path.basename(file).replace("_iou_comparison.csv", "")
        df["video_id"] = video_id
    
    all_dfs.append(df)

# Concatenate into one dataframe
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save
combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Combined {len(csv_files)} files into {OUTPUT_FILE}")
print(combined_df.columns)