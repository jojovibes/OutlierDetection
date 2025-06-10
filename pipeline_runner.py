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

ROOT_DIR = '/home/jlin1/OutlierDetection/testing/frames'
output_folder = "output_only_logits"

OUTPUT_DIR = os.path.join(ROOT_DIR, output_folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(ROOT_DIR, "completed_folders.log")

def main():
    for folder_name in os.listdir(ROOT_DIR):
        if folder_name == output_folder:
            continue

        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
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

