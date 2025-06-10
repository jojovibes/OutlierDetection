# import pandas as pd
# from sklearn.mixture import GaussianMixture
# from utilz import select_feature_columns
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np


# def run(df):

#     X = df[select_feature_columns(df)]
#     n_samples = len(X)

#     if n_samples < 2:
#         print(f"[GMM] Not enough samples for GMM (n={n_samples}). Returning NaNs.")
#         return pd.Series([np.nan] * n_samples, index=df.index, name='score_gmm')

#     n_components = min(4, n_samples)
 
# #
#     try:
#         gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
#         gmm.fit(X)
#         scores = gmm.score_samples(X)
#         anomaly_score = -scores
#         score_normalized = MinMaxScaler().fit_transform(anomaly_score.reshape(-1, 1)).flatten()

#         return pd.Series(score_normalized, index=df.index, name='score_gmm')   # lower = more anomalous

#     except Exception as e:
#         print(f"[GMM] Error fitting GMM: {e}")
#         return pd.Series([np.nan] * n_samples, index=df.index, name='score_gmm')

import pandas as pd
from sklearn.mixture import GaussianMixture
from utilz import select_feature_columns
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def run(df):
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
