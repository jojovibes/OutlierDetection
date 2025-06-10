# import pandas as pd
# from cadi.Src.forest import Forest
# from cadi.Src.dataset import *
# from utilz import select_feature_columns
# import tempfile
# import numpy as np

# def run(df):

#     NB_TREES = 100
#     MAX_HEIGHT = 1000
#     CONTAMINATION_RATE = 0.05

#     feature_cols = select_feature_columns(df)
#     X = df[feature_cols]

#     with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmpfile:
#         X.to_csv(tmpfile.name, index=False, header=False)
#         dataset = Dataset(tmpfile.name) 

#     f = Forest(dataset, nbT=NB_TREES, method="cadi", maxHeight=MAX_HEIGHT)
#     f.build()

#     f.anomalyDetection(binary=True, contamination_rate=CONTAMINATION_RATE)
#     f.clustering()
#     f.explain_anomalies()

#     # df = pd.DataFrame(dataset.data)

#     df = df.reset_index(drop=True)

#     # is_anomaly = np.zeros(len(df), dtype=int)
#     # is_anomaly[f.anomalies] = 1

#     # print(df.describe())

#     # df["cadi_anomaly"] = is_anomaly
#     df["cadi_anomaly"] = 0
#     df.loc[f.anomalies, "cadi_anomaly"] = 1
#     df["score_cadi"] = f.scores
#     df["cadi_cluster"] = f.clusters_affectations
#     df["cadi_explanation"] = [f.explanations.get(i, "") for i in range(len(df))]


#     return df

import pandas as pd
import numpy as np

import tempfile
from cadi.Src.forest import Forest
from cadi.Src.dataset import Dataset
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GaussianMixture
from utilz import select_feature_columns
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble import IsolationForest
from utilz import select_feature_columns



def run_cadi(df):
    NB_TREES = 100
    MAX_HEIGHT = 100

    # context_features = ['frame_idx', 'class_id'] + class_prob_features
    # behavior_features = ['velocity', 'direction', 'width', 'height', 'center_x', 'center_y', 'confidence']
    context_features = ['class_id','center_x', 'center_y','width', 'height', 'ratio', 'area']
    behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)]  # assuming 12 relevant classes
    # behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)] + [f'class_prob_{i}' for i in range(12)] # assuming 12 relevant classes

    df_out = []

    # Group by context — in this example: class_id
    grouped = df.groupby("class_id")

    for class_id, group_df in grouped:
        if len(group_df) < 10:
            continue  # too small to isolate

        try:
            X_c = StandardScaler().fit_transform(group_df[context_features])
            X_b = StandardScaler().fit_transform(group_df[behavior_features])
            X_combined = np.hstack([X_c, X_b])

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmpfile:
                pd.DataFrame(X_combined).to_csv(tmpfile.name, index=False, header=False)
                dataset = Dataset(tmpfile.name)

            f = Forest(dataset, nbT=NB_TREES, method="cadi", maxHeight=MAX_HEIGHT)
            f.build()
            f.anomalyDetection(binary=False)
            f.clustering()
            f.explain_anomalies()

            group_df = group_df.reset_index(drop=True)
            group_df["score_cadi"] = f.scores
            group_df["cadi_anomaly"] = 0
            group_df.loc[f.anomalies, "cadi_anomaly"] = 1
            group_df["cadi_cluster"] = f.clusters_affectations
            group_df["cadi_explanation"] = [f.explanations.get(i, "") for i in range(len(group_df))]

            df_out.append(group_df)

        except Exception as e:
            print(f"[CADI] Error processing class_id={class_id}: {e}")

    return pd.concat(df_out, ignore_index=True)



def run_GMM(df):
    X = df[select_feature_columns(df)]
    n_samples = len(X)

    if n_samples < 2:
        return pd.Series([np.nan] * n_samples, index=df.index, name='score_gmm')

    try:
        # Clean and normalize input
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        X_scaled = MinMaxScaler().fit_transform(X)

        # Tune number of components using BIC
        best_bic = np.inf
        best_gmm = None
        for k in range(1, min(6, n_samples)):
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        # Anomaly scores (negative log-likelihood)
        scores = best_gmm.score_samples(X_scaled)
        anomaly_score = -scores
        score_normalized = MinMaxScaler().fit_transform(anomaly_score.reshape(-1, 1)).flatten()


        return pd.Series(score_normalized, index=df.index, name='score_gmm')

    except Exception as e:
        print(f"[GMM] Error fitting GMM: {e}")
        return pd.Series([np.nan] * n_samples, index=df.index, name='score_gmm')


def run_IF(df):

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
