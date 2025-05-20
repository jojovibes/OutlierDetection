import pandas as pd
from sklearn.mixture import GaussianMixture
from utilz import select_feature_columns


def run(df):

    X = df[select_feature_columns(df)]

    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
    gmm.fit(X)

    log_likelihoods = gmm.score_samples(X)

    return pd.Series(-log_likelihoods, index=df.index, name='score_gmm')   # lower = more anomalous

