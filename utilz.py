import pandas as pd

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

    # features = ["center_x", "center_y", "width", "height"]

    return df