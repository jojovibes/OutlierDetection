import json
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

metadata_file_path = "/Volumes/ronni/shanghaitech/metadata_output/01_0014_metadata.json"
output_path = "outputs/gmm_scored_metadata.csv"

with open(metadata_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)

df["width"] = df["x2"] - df["x1"]
df["height"] = df["y2"] - df["y1"]
df["center_x"] = (df["x1"] + df["x2"]) / 2
df["center_y"] = (df["y1"] + df["y2"]) / 2

features = ["center_x", "center_y", "width", "height", "velocity", "direction", "class_id", "confidence"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
gmm.fit(X)

log_likelihoods = gmm.score_samples(X)
df["anomaly_score"] = -log_likelihoods  # lower = more anomalous

df.to_csv(output_path, index=False)
print(f"Saved scored metadata to: {output_path}")
