from cadi.Src.forest import Forest
from cadi.Src.dataset import *
import pandas as pd
import numpy as np
import tempfile
from utilz import select_feature_columns
from cadi.Src.viewer import viewDatasetWithAnomalies
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score)

# # d = Dataset("cadi/Data/cardio.csv", labels=False)
# #open
# df = pd.read_csv("/home/jlin1/OutlierDetection/cadi/Data/arrhythmia.csv")

# # df_numeric = df.select_dtypes(include=[float, int])  # or manually drop: df.drop(columns=["class_id", "track_id"], inplace=True)
# # data = df_numeric.values

# feature_cols = select_feature_columns(df)
# data = df[feature_cols]

# with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmpfile:
#     data.to_csv(tmpfile.name, index=False, header=False)
#     dataset = Dataset(tmpfile.name) 


# nbT = 100
# f = Forest(dataset, nbT, "cadi", maxHeight=100)
# f.build()

# f.anomalyDetection(binary=False)
# f.clustering()
# f.explain_anomalies()

# print(len(f.anomalies), "out of", len(f.scores))

# df_result = pd.DataFrame(dataset.data)

# print("Score stats - min:", np.min(f.scores), "max:", np.max(f.scores))
# print("Number of anomalies:", len(f.anomalies), "/", len(f.scores))
# print(f.anomalies)


# df_result["cadi_anomaly"] = 0
# df_result.loc[f.anomalies, "cadi_anomaly"] = 1
# df_result["cadi_score"] = f.scores
# df_result["cadi_cluster"] = f.clusters_affectations
# df_result["cadi_explanation"] = [f.explanations.get(i, {}) for i in range(len(df_result))]

# # Save output
# df_result.to_csv("cadi_output.csv", index=False)

# # anomalies = f.anomalies
# # explanations = f.explanations  # Optional
# th = np.percentile(f.scores, 90)   

# # Call the function
# viewDatasetWithAnomalies(dataset, f.scores, ALPHA=th)
# plt.savefig("cadi_anomalies_plot.png", dpi=300, bbox_inches="tight")
# plt.close()

import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

data = pd.read_csv('/home/jlin1/OutlierDetection/cadi/Data/arrhythmia.csv')
# data = pd.read_csv('/home/jlin1/OutlierDetection/cadi/Data/mammography.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values  # Last column: 1 for anomaly, 0 for normal

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmpfile:
    X.to_csv(tmpfile.name, index=False, header=False)
    dataset = Dataset(tmpfile.name) 

n_runs = 10
nbT = 100
auc_scores = []
ap_scores =[]


for _ in range(n_runs):
    # Fit the model
    f = Forest(dataset, nbT, "cadi", maxHeight=8)
    f.build()
    f.anomalyDetection(binary=True, contamination_rate=0.05)
    f.clustering()

    scores = f.scores

    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y,scores)
    auc_scores.append(auc)
    ap_scores.append(ap)

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
print(f"Mean AP: {mean_ap:.3f} ± {std_ap:.3f}")
