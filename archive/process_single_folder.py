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
from utilz import derive_features
from outlierDetection import run_cadi, run_GMM, run_IF

ROOT_DIR = '/home/jlin1/OutlierDetection/UCSDped2/frames'
OUTPUT_DIR = os.path.join(ROOT_DIR, "preprocess_fix")

def process_folder(folder_name):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    data = []
    prev_bboxes = {}

    for fname in sorted(os.listdir(folder_path)):
        if fname.startswith("._") or not fname.endswith(".jpg"):
            continue

        fpath = os.path.join(folder_path, fname)

        try:
            frame_idx = int(os.path.splitext(fname)[0])
            # if frame_idx == 30:
            #     break

            print(f"fram_idx: {frame_idx}")

            img = cv2.imread(fpath)
            if img is None:
                print(f"[Frame {frame_idx}] Warning: image not readable")
                continue

            features_list = extract_features(img, prev_bboxes, frame_idx=frame_idx) #preprocesing

            if not features_list:
                print(f"[Frame {frame_idx}] Warning: no features extracted")
                continue

            for features in features_list:
                features['filename'] = fname
                features['frame_idx'] = frame_idx
                data.append(features)

            del img, features_list
            gc.collect()

        except Exception as e:
            import traceback
            print(f"[Frame {frame_idx}] Error: {fname}\n{traceback.format_exc()}")

    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return

    features_path = os.path.join(OUTPUT_DIR, f"{folder_name}_features.csv")
    scored_path = os.path.join(OUTPUT_DIR, f"{folder_name}_scored.csv")
    

    df = derive_features(df) #derive features utils

    df.drop(columns=['class_probabilities'], inplace=True, errors='ignore')

    # df = df[df['class_probabilities'].apply(lambda x: isinstance(x, list) and all(np.isfinite(x)))].reset_index(drop=True)

    # class_probs = pd.DataFrame(df.pop('class_probabilities').tolist(), index=df.index)
    # class_probs.columns = [f'class_prob_{i}' for i in range(class_probs.shape[1])]
    # df[class_probs.columns] = class_probs

    df = df[df['logits'].apply(lambda x: isinstance(x, list) and all(np.isfinite(x)))].reset_index(drop=True)

    logits = pd.DataFrame(df.pop('logits').tolist(), index=df.index)
    # logits.columns = [f'logit_{i}' for i in range(class_probs.shape[1])]
    logits.columns = [f'logit_{i}' for i in range(len(logits.columns))]
    df[logits.columns] = logits

    original_bbox = df[['bbox']].copy()
    df.drop(columns=['bbox'], inplace=True, errors='ignore')
    # 'x1', 'y1', 'x2', 'y2'

    exclude_cols = ['track_id', 'filename', 'frame_idx']
    scale_cols = [col for col in df.columns if col not in exclude_cols]
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    df.to_csv(features_path, index=False)

    try:
        df['score_if'] = run_IF(df)
        df['score_gmm'] = run_GMM(df)
        df = run_cadi(df)
        df['score_avg'] = (df['score_gmm'] + df['score_cadi']) / 2

    except Exception as e:
        print(f"Scoring failed: {e}")
        return

    df = pd.concat([df, original_bbox], axis=1)
    df = df.copy() 

    df.to_csv(scored_path, index=False)
    print(f"{folder_name}_scored.csv has been saved")


    if os.path.exists(scored_path):
        try:
            os.remove(features_path)
            print(f"[{folder_name}] Done — features file removed.")
        except Exception as e:
            print(f"[{folder_name}] Scored saved but failed to delete features file: {e}")
    else:
        print(f"[{folder_name}] Warning: scored CSV not saved, features retained.")

if __name__ == "__main__":
    folder = sys.argv[1]
    process_folder(folder)
