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


def run(df):
    NB_TREES = 100
    MAX_HEIGHT = 100

    # Define context and behavior features
    class_prob_features = [f'class_prob_{i}' for i in range(80)]
    # context_features = ['frame_idx', 'class_id'] + class_prob_features
    # behavior_features = ['velocity', 'direction', 'width', 'height', 'center_x', 'center_y', 'confidence']
    context_features = ['class_id','center_x', 'center_y','width', 'height', 'ratio', 'area']
    behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)]  # assuming 12 relevant classes


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

    print("finished cadi")
    return pd.concat(df_out, ignore_index=True)
