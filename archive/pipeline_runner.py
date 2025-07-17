import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import gc
import psutil
import subprocess
import traceback
# np.float = float  

ROOT_DIR = '/home/jlin1/OutlierDetection/UCSDped2/frames'
output_folder = "preprocess_fix"

OUTPUT_DIR = os.path.join(ROOT_DIR, output_folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(ROOT_DIR, "completed_folders.log")

def main():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            completed = set(line.strip() for line in f if line.strip())
    else:
        completed = set()

    for folder_name in os.listdir(ROOT_DIR):
        if folder_name == output_folder:
            continue

        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        if folder_name in completed:
            print(f"[{folder_name}] Already processed, skipping.")
            continue
        
        print(f"Processing folder {folder_name} in a subprocess...")

        try:
            subprocess.run(["python", "process_single_folder.py", folder_name], check=True) #scoring

            with open(LOG_FILE, "a") as f:
                f.write(folder_name + "\n")
            print(f"[{folder_name}] Logged as completed.")

        except subprocess.CalledProcessError as e:
            print(f"[{folder_name}] crashed with exit code {e.returncode}, skipping.")

if __name__ == "__main__":
    main()

