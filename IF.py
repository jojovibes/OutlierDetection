import json
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

metadata_file_path = "/Volumes/ronni/shanghaitech/metadata_output/01_0014_metadata.json"
output_path = "outputs/isoforest_scored_metadata.csv"

# Load metadata
with open(metadata_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Expand bounding box coordinates
df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
df["width"] = df["x2"] - df["x1"]
df["height"] = df["y2"] - df["y1"]
df["center_x"] = (df["x1"] + df["x2"]) / 2
df["center_y"] = (df["y1"] + df["y2"]) / 2

# Feature extraction and normalization
features = ["center_x", "center_y", "width", "height", "velocity", "direction", "class_id", "confidence"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Fit Isolation Forest
isoforest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
isoforest.fit(X)

# Anomaly score (negative = more anomalous)
df["anomaly_score"] = -isoforest.score_samples(X)

# Save result
df.to_csv(output_path, index=False)
print(f"Saved scored metadata to: {output_path}")
