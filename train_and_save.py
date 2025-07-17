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

import joblib
import pickle

def train_and_save_IF(df, save_path="models/isoforest.pkl"):
    X = df[select_feature_columns(df)]
    
    isoforest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    isoforest.fit(X)

    joblib.dump(isoforest, save_path)
    print("Train IF completed")
    return isoforest

def load_and_run_IF(df, model_path="models/isoforest.pkl"):
    isoforest = joblib.load(model_path)

    X = df[select_feature_columns(df)]
    scores = -isoforest.score_samples(X)
    print("Run IF completed")

    return pd.Series(scores, index=df.index, name='score_if')

def train_and_save_GMM(df, save_path="models/gmm.pkl"):
    X = df[select_feature_columns(df)]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X_scaled = MinMaxScaler().fit_transform(X)

    best_bic = np.inf
    best_gmm = None
    for k in range(1, min(6, len(X))):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    joblib.dump(best_gmm, save_path)

    with open("models/gmm_features.pkl", "wb") as f_feat:
        pickle.dump(X.columns.tolist(), f_feat)
    
    print("Train GMM completed")

    return best_gmm

def load_and_run_GMM(df, model_path="models/gmm.pkl"):
    gmm = joblib.load(model_path)

    with open("models/gmm_features.pkl", "rb") as f_feat:
        feature_cols = pickle.load(f_feat)

    X = df[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X_scaled = MinMaxScaler().fit_transform(X)

    scores = -gmm.score_samples(X_scaled)
    score_normalized = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()

    print("Run GMM completed")

    return pd.Series(score_normalized, index=df.index, name='score_gmm')

def train_and_save_cadi(df):
    NB_TREES = 100
    MAX_HEIGHT = 100

    # context_features = ['frame_idx', 'class_id'] + class_prob_features
    # behavior_features = ['velocity', 'direction', 'width', 'height', 'center_x', 'center_y', 'confidence']
    context_features = ['class_id','center_x', 'center_y','width', 'height', 'ratio', 'area']
    behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)]  # assuming 12 relevant classes
    # behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)] + [f'class_prob_{i}' for i in range(12)] # assuming 12 relevant classes

    df_out = []

    grouped = df.groupby("class_id")

    for class_id, group_df in grouped:
        if len(group_df) < 10:
            continue  
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

        with open("models/cadi.pkl", "wb") as f_out:
            pickle.dump(f, f_out)

        print("Train CADI completed")
    
def load_and_run_CADI(df, model_path="models/cadi.pkl"):
    with open(model_path, "rb") as f_in:
        forest = pickle.load(f_in)

    context_features = ['class_id','center_x', 'center_y','width', 'height', 'ratio', 'area']
    behavior_features = ['velocity', 'direction', 'confidence'] + [f'logit_{i}' for i in range(12)]

    df_out = []

    grouped = df.groupby("class_id")

    for class_id, group_df in grouped:
        if len(group_df) < 10:
            continue

        try:
            X_c = StandardScaler().fit_transform(group_df[context_features])
            X_b = StandardScaler().fit_transform(group_df[behavior_features])
            X_combined = np.hstack([X_c, X_b])

            scores = np.apply_along_axis(forest.computeScore, 1, X_combined)
            # scores = forest.score_data(X_combined)  
            group_df["score_cadi"] = scores

            df_out.append(group_df)

        except Exception as e:
            print(f"[CADI] Error on class_id={class_id}: {e}")
        
    print("Run CADI completed")

    return pd.concat(df_out, ignore_index=True)
