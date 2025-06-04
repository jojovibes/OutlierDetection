import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import gc
import psutil
import subprocess
import traceback


np.float = float  
#v

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
            torch.cuda.empty_cache()  
            gc.collect()

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return pd.DataFrame(data)

LOG_FILE = os.path.join(ROOT_DIR, "completed_folders.log")


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

if __name__ == "__main__":
    main()

