import pandas as pd
from cadi.Src.forest import Forest
from cadi.Src.dataset import *
from utilz import select_feature_columns
import tempfile

def run(df):

    NB_TREES = 100
    MAX_HEIGHT = 256
    CONTAMINATION_RATE = 0.05

    feature_cols = select_feature_columns(df)
    X = df[feature_cols]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmpfile:
        X.to_csv(tmpfile.name, index=False, header=False)
        dataset = Dataset(tmpfile.name) 

    f = Forest(dataset, nbT=NB_TREES, method="cadi", maxHeight=MAX_HEIGHT)
    f.build()

    f.anomalyDetection(binary=True, contamination_rate=CONTAMINATION_RATE)
    f.clustering()
    f.explain_anomalies()

    # df = pd.DataFrame(dataset.data, columns=feature_cols)

    is_anomaly = np.zeros(len(df), dtype=int)
    is_anomaly[f.anomalies] = 1

    df["cadi_anomaly"] = is_anomaly
    df["score_cadi"] = f.scores
    df["cadi_cluster"] = f.clusters_affectations
    df["cadi_explanation"] = f.explanations

    return df

    # X = df[select_feature_columns(df)].to_numpy()

    # dataset = Dataset(X)

    # f = Forest(dataset, nbT=NB_TREES, method="cadi", maxHeight=MAX_HEIGHT)
    # f.build()

    # f.anomalyDetection(binary=True, contamination_rate=CONTAMINATION_RATE)
    # f.clustering()
    # f.explain_anomalies()

    # feature_cols = select_feature_columns(df)
    # df = pd.DataFrame(dataset.data, columns=feature_cols)

    # is_anomaly = np.zeros(len(df), dtype=int)
    # is_anomaly[f.anomalies] = 1

    # df["cadi_anomaly"] = is_anomaly
    # df["cadi_score"] = f.scores
    # df["cadi_cluster"] = f.clusters_affectations # cluster labels (-1 for anomalies)
    # df["cadi_explanation"] = [f.explanations.get(i, "") for i in range(len(df))]

    # return df


    # os.makedirs("outputs", exist_ok=True)
    # df.to_csv(OUTPUT_PATH, index=False)
    # print(f"output path: {OUTPUT_PATH}")
