import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import gc
import psutil
import cv2

np.float = float

from preprocess import extract_features
from GMM import run as run_GMM
from IF import run as run_IF
from utilz import derive_features
from outlierDetection import run as run_cadi

ROOT_DIR = '/home/jlin1/OutlierDetection/testing/small_batch'
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

def log_mem(stage):
    print(f"[{stage}] Memory used: {psutil.virtual_memory().used / 1e9:.2f} GB")

def process_folder(folder_name):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    data = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.startswith("._") or not fname.endswith(".jpg"):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            frame_idx = int(os.path.splitext(fname)[0])
            if frame_idx == 30:
                break

            img = cv2.imread(fpath)
            if img is None:
                continue

            features_list = extract_features(img)
            if not features_list:
                continue

            for features in features_list:
                features['filename'] = fname
                features['frame_idx'] = frame_idx
                data.append(features)

            del img, features_list
            gc.collect()

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return

    log_mem("After DataFrame")

    df = derive_features(df)
    class_probs = pd.DataFrame(df.pop('class_probabilities').tolist(), index=df.index)
    class_probs.columns = [f'class_prob_{i}' for i in range(class_probs.shape[1])]
    df[class_probs.columns] = class_probs
    df.drop(columns=['bbox', 'x1', 'y1', 'x2', 'y2'], inplace=True, errors='ignore')

    exclude_cols = ['track_id', 'filename', 'frame_idx']
    scale_cols = [col for col in df.columns if col not in exclude_cols]
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    df.to_csv(os.path.join(OUTPUT_DIR, f"{folder_name}_features.csv"), index=False)

    # Uncomment scoring below as needed:
    # df['score_if'] = run_IF(df)
    # df['score_gmm'] = run_GMM(df)
    # df = run_cadi(df)
    # df['score_avg'] = (df['score_gmm'] + df['score_cadi']) / 2

    df.to_csv(os.path.join(OUTPUT_DIR, f"{folder_name}_scored.csv"), index=False)
    print(f"[{folder_name}] Done")

if __name__ == "__main__":
    folder = sys.argv[1]
    process_folder(folder)
