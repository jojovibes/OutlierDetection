import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def apply_platt_scaling(df, score_col):

    X = df[[score_col]].values 
    y = df["mask_anomaly"].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)


    df[f"{score_col}_platt"] = model.predict_proba(X)[:, 1]
    return df, model


def select_feature_columns(df):
    exclude_cols = ['filename', 'track_id', 'frame_idx', 'bbox', 'class_probabilities']
    return [
        col for col in df.columns
        if col not in exclude_cols and df[col].apply(lambda x: isinstance(x, (int, float))).all()
    ]

def derive_features(df):
    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
    df["width"] = df["x2"] - df["x1"]
    df["height"] = df["y2"] - df["y1"]
    df["center_x"] = (df["x1"] + df["x2"]) / 2
    df["center_y"] = (df["y1"] + df["y2"]) / 2
    df["ratio"] = df["height"] / df["width"]
    df["area"] = df["height"] * df["width"]

    # features = ["center_x", "center_y", "width", "height"]

    return df