from cadi.Src.forest import Forest
from cadi.Src.dataset import *
import pandas as pd
import numpy as np

d = Dataset("cadi/Data/cardio.csv", labels=False)

nbT = 100
f = Forest(d, nbT, "cadi", maxHeight=256)
f.build()

f.anomalyDetection(binary=True, contamination_rate=0.05)
f.clustering()
print(f.clusters_affectations) 
f.explain_anomalies()
print(f.explanations) 


df = pd.DataFrame(d.data)

is_anomaly = np.zeros(len(df), dtype=int)
is_anomaly[f.anomalies] = 1
df["cadi_anomaly"] = is_anomaly

df["cadi_score"] = f.scores
df["cadi_cluster"] = f.clusters_affectations

df["cadi_explanation"] = [f.explanations.get(i, {}) for i in range(len(df))]

df.to_csv("cadi_output.csv", index=False)