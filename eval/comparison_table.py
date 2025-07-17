import pandas as pd
import os

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
# OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results_detection")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing"
# SCORE_DIR = os.path.join(ROOT_DIR, "frames/output_only_logits")
SCORE_DIR = "/home/jlin1/OutlierDetection/UCSDped2/frames/output_only_logits"
# MASKS_DIR = "/home/jlin1/OutlierDetection/testing/test_pixel_mask"
# MASKS_DIR = "/home/jlin1/OutlierDetection/UCSDped2/mask"
OUTPUT_DIR = os.path.join(SCORE_DIR, "comparison_results_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_metrics_path = os.path.join("/home/jlin1/OutlierDetection/UCSDped2/frames/output_only_logits/comparison_results_frame/frame_metrics.csv")
detection_metrics_path = os.path.join("/home/jlin1/OutlierDetection/UCSDped2/frames/output_only_logits/comparison_results_detection/performance_metrics.csv")

df_frame = pd.read_csv(frame_metrics_path)
df_detect = pd.read_csv(detection_metrics_path)

df_frame.rename(columns={"Method": "Model"}, inplace=True)
df_detect.rename(columns={"Method": "Model"}, inplace=True)

df_merged = pd.merge(df_frame, df_detect, on="Model", how="outer")

df_final = df_merged.copy()

col_order = ['Model'] + [col for col in df_final.columns if col not in ['Model', 'RBDC']] + ['RBDC']
df_final = df_final[col_order]

df_final.set_index("Model", inplace=True)
print(df_final)
df_final.to_csv(os.path.join(OUTPUT_DIR, "combined_metrics_summary.csv"))
