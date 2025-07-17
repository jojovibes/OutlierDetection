# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     roc_curve,
#     auc
# )

# ROOT_DIR = "/home/jlin1/OutlierDetection/testing/small_batch/output/score/comparison_results/01_0054_iou_comparison.csv"
# SAVE_DIR = "figures"                  
# os.makedirs(SAVE_DIR, exist_ok=True)  
# df = pd.read_csv("/home/jlin1/OutlierDetection/testing/small_batch/output/comparison_results/01_0063_iou_comparison.csv") 


# df['cadi_anomaly'] = df['cadi_anomaly'].astype(int)
# df['mask_anomaly'] = df['mask_anomaly'].astype(int)

# print(df['cadi_anomaly'].value_counts()) 
# print(pd.crosstab(df['mask_anomaly'], df['cadi_anomaly'], rownames=['True'], colnames=['Predicted'])
# )


# y_true = df['mask_anomaly']
# y_pred = df['cadi_anomaly']

# precision = precision_score(y_true, y_pred, zero_division=0)
# recall = recall_score(y_true, y_pred, zero_division=0)
# f1 = f1_score(y_true, y_pred, zero_division=0)

# print("=== Classification Metrics ===")
# print(f"Precision: {precision:.3f}")
# print(f"Recall:    {recall:.3f}")
# print(f"F1-Score:  {f1:.3f}")

# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
# disp.plot(cmap="Blues")
# plt.title("Confusion Matrix")
# conf_mat_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
# plt.savefig(conf_mat_path)
# print(f"[Saved] Confusion matrix → {conf_mat_path}")
# plt.close()

# #check with y_pred
# if 'iou' in df.columns and df['iou'].nunique() > 1:
#     y_scores = df['iou']
# else:
#     y_scores = y_pred  

# fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# roc_path = os.path.join(SAVE_DIR, "roc_curve.png")
# plt.savefig(roc_path)
# print(f"[Saved] ROC curve → {roc_path}")
# plt.close()

# # Test flipping predictions
# df['cadi_anomaly_flipped'] = 1 - df['cadi_anomaly']

# from sklearn.metrics import classification_report

# print("=== Evaluation after flipping ===")
# print(pd.crosstab(df['mask_anomaly'], df['cadi_anomaly_flipped'], rownames=['True'], colnames=['Predicted']))
# print(classification_report(df['mask_anomaly'], df['cadi_anomaly_flipped'], target_names=['Normal', 'Anomaly']))

