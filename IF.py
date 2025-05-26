import pandas as pd
from sklearn.ensemble import IsolationForest
from utilz import select_feature_columns
import numpy as np

def run(df):

    X = df[select_feature_columns(df)]

    isoforest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    isoforest.fit(X)

    # Anomaly score (negative = more anomalous)
    scores = -isoforest.score_samples(X)
    # df["anomaly_score"] = -isoforest.score_samples(df)

    return pd.Series(scores, index=df.index, name='score_if')

# metadata_file_path = "/Volumes/ronni/shanghaitech/metadata_output/01_0014_metadata.json"
# output_path = "outputs/isoforest_scored_metadata.csv"

# df.to_csv(output_path, index=False)
# print(f"Saved scored metadata to: {output_path}")
