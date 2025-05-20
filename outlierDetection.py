import os
import pandas as pd
from cadi.Src.forest import Forest
from cadi.Src.dataset import *

DATA_PATH = "cadi/Data/cardio.csv"
OUTPUT_PATH = "outputs/cardio_with_cadi.csv"
NB_TREES = 100
MAX_HEIGHT = 256
CONTAMINATION_RATE = 0.05

data = Dataset(DATA_PATH, labels=False)

f = Forest(data, nbT=NB_TREES, method="cadi", maxHeight=MAX_HEIGHT)
f.build()

f.anomalyDetection(binary=True, contamination_rate=CONTAMINATION_RATE)
f.clustering()
f.explain_anomalies()

df = pd.DataFrame(d.data, columns=d.attributes)


df["cadi_score"] = f.scores
df["cadi_cluster"] = f.clusters_affectations # cluster labels (-1 for anomalies)
df["cadi_explanation"] = f.explanations


os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"output path: {OUTPUT_PATH}")
