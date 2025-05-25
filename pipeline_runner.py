import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import gc
import psutil
import subprocess
import traceback


np.float = float  # TEMP fix for deprecated np.float

from preprocess import extract_features
from GMM import run as run_GMM
from IF import run as run_IF
from utilz import derive_features
from outlierDetection import run as run_cadi

ROOT_DIR = '/home/jlin1/OutlierDetection/testing/small_batch'
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_folder(folder_path):
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

            del img
            del features_list
            torch.cuda.empty_cache()  # Only if using CUDA
            gc.collect()

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return pd.DataFrame(data)

LOG_FILE = os.path.join(ROOT_DIR, "completed_folders_1.log")
print(f"logging path: {ROOT_DIR}")

def main():
    for folder_name in os.listdir(ROOT_DIR):
        if folder_name == "output":
            continue

        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder {folder_name} in a subprocess...")

        try:
            subprocess.run(["python", "process_single_folder.py", folder_name], check=True)

            with open(LOG_FILE, "a") as f:
                f.write(folder_name + "\n")
            print(f"[{folder_name}] Logged as completed.")

        except subprocess.CalledProcessError as e:
            print(f"[{folder_name}] crashed with exit code {e.returncode}, skipping.")



# def main():
#     for folder_name in os.listdir(ROOT_DIR):
#         if folder_name == "output":
#             continue

#         folder_path = os.path.join(ROOT_DIR, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
        
#         print(f"Processing folder {folder_name} in a subprocess...")

#         try:
#             subprocess.run(["python", "process_single_folder.py", folder_name], check=True)

#             with open(LOG_FILE, "a") as f:
#                     f.write(folder_name + "\n")
#                 print(f"[{folder_name}] Logged as completed.")

#         except subprocess.CalledProcessError:
#             print(f"[{folder_name}] Failed — not logged.")

if __name__ == "__main__":
    main()

# def main():
#     for folder_name in os.listdir(ROOT_DIR):
#         if folder_name == "output":
#             continue

#         folder_path = os.path.join(ROOT_DIR, folder_name)
#         if not os.path.isdir(folder_path):
#             continue

#         log_mem("Start")

#         df = process_folder(folder_path)
#         log_mem("After process_folder")

#         if df.empty:
#             print(f"No features extracted from {folder_name}")
#             continue

#         df = derive_features(df)
#         log_mem("After derive_features")

#         # Avoid extra copy: directly assign expanded columns
#         class_probs = pd.DataFrame(df.pop('class_probabilities').tolist(), index=df.index)
#         class_probs.columns = [f'class_prob_{i}' for i in range(class_probs.shape[1])]
#         df[class_probs.columns] = class_probs
#         del class_probs
#         gc.collect()
#         log_mem("After expanding class_probs")

#         df.drop(columns=['bbox', 'x1', 'y1', 'x2', 'y2'], inplace=True, errors='ignore')
#         log_mem("After drop bbox/coords")

#         exclude_cols = ['track_id', 'filename', 'frame_idx']
#         scale_cols = [col for col in df.columns if col not in exclude_cols]

#         scaler = StandardScaler()
#         df[scale_cols] = scaler.fit_transform(df[scale_cols])
#         log_mem("After scaling")

#         csv_path = os.path.join(OUTPUT_DIR, f"{folder_name}_features.csv")
#         df.to_csv(csv_path, index=False)
#         log_mem("Saved feature CSV")

#         # Uncomment these one by one and watch memory logs
#         # df['score_if'] = run_IF(df)
#         # log_mem("After IF")

#         # df['score_gmm'] = run_GMM(df)
#         # log_mem("After GMM")

#         # df = run_cadi(df)
#         # log_mem("After CADI")

#         # df['score_avg'] = (df['score_gmm'] + df['score_cadi']) / 2

#         out_path = os.path.join(OUTPUT_DIR, f"{folder_name}_scored.csv")
#         df.to_csv(out_path, index=False)
#         log_mem("Saved scored CSV")

#         del df, folder_path, folder_name
#         gc.collect()
#         log_mem("After GC for folder")