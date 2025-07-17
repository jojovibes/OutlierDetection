from pathlib import Path
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import ast
import numpy as np

import random
import os
import tempfile
import shutil

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "VAD" / "src"))

from vad.data_shanghai import index_shanghaitech_files
from vad.extract_shanghai import run_shanghai_extract
from vad.util import load_db
# from outlierDetection import run_cadi, run_IF, run_GMM
from train_and_save import train_and_save_IF, train_and_save_GMM, train_and_save_cadi, load_and_run_IF, load_and_run_GMM, load_and_run_CADI

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None 

# MODE = "training"  
MODE = "testing"

if MODE == "training":
    ROOT_DIR = Path("/home/jlin1/OutlierDetection/shanghaitech(og)/training/videos/frames")
    DATASET_NAME = "shanghai_train"
    OUT_DIR = Path("outputs/shanghai_train")
    MODEL_DIR = Path("models/shanghai_train")
else:
    ROOT_DIR = Path("/home/jlin1/OutlierDetection/shanghaitech(og)/testing/frames")
    DATASET_NAME = "shanghai_test"
    OUT_DIR = Path("outputs/shanghai_test")
    MODEL_DIR = Path("models/shanghai_train")  # ← reuse trained models from training set

DB_NAME = f"{DATASET_NAME}.db"
DB_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# if MODE == "training":
#     num_of_sample = 11
#     random.seed(42)
#     all_folders = [f for f in os.listdir(ROOT_DIR)
#                 if os.path.isdir(os.path.join(ROOT_DIR, f))]


#     selected_folders = random.sample(all_folders, num_of_sample)

#     selected_paths = [os.path.join(ROOT_DIR, f) for f in selected_folders]
#     print(selected_folders)

#     TEMP_ROOT_DIR = Path("temp_sampled_train_set")
#     if TEMP_ROOT_DIR.exists():
#         shutil.rmtree(TEMP_ROOT_DIR)

#     TEMP_ROOT_DIR.mkdir(parents=True)

#     for folder in selected_folders:
#         src = ROOT_DIR / folder
#         dst = TEMP_ROOT_DIR / folder
#         os.symlink(src, dst)

#     ROOT_DIR = TEMP_ROOT_DIR  #


# # Index all frames across all folders into one DB
# index_shanghaitech_files(DATASET_NAME, MODE, ROOT_DIR)

# print("step 1 complete")

# # # Run extract (YOLOv7 + BoT-SORT + feature tracking)
# run_shanghai_extract(ROOT_DIR,OUT_DIR, DB_NAME)
# # run_extract(ROOT_DIR, OUT_DIR, DB_NAME, MODE)
# print("step 2 complete")

# # Load features from DB
db_path = OUT_DIR / f"{DATASET_NAME}.db" # still named this inside extract
# df = load_db(db_path, "features")
df = load_db(db_path, "features")

print(df.head)

df.to_csv("cleaned_data.csv", index=False)

# print("complete step 4")

# # Derive and scale features
# original_bbox = df[["x1", "y1", "x2", "y2"]].deepcopy() #pointer check
# exclude_cols = ["track_id", "frame_idx", "frame_file", "frame_dir", "path", "class_probabilities", "class_id", "confidence"]
# # scale_cols = [col for col in df.columns if col not in exclude_cols and isinstance(df[col].iloc[0], (int, float, float))]

# numeric_cols = df.select_dtypes(include='number').columns.tolist()
# scale_cols = [col for col in numeric_cols if col not in exclude_cols]

# df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

# # Save raw features
# features_path = OUT_DIR / "all_features.csv"
# df.to_csv(features_path, index=False)

if MODE == "training":
    train_and_save_IF(df)
    train_and_save_GMM(df)
    train_and_save_cadi(df)

if MODE == "testing":
    df["score_if"] = load_and_run_IF(df)
    df["score_gmm"] = load_and_run_GMM(df)
    df = load_and_run_CADI(df)  

    score_cols = ["score_if", "score_gmm", "score_cadi"]

    # Min-max normalization per column
    # for col in score_cols:
    #     min_val = df[col].min()
    #     max_val = df[col].max()
    #     df[col + "_norm"] = (df[col] - min_val) / (max_val - min_val)


    for col in score_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val

        df[col + "_qnorm"] = df[col].rank(method="average", pct=True)

        if range_val > 0:
            df[col + "_norm"] = (df[col] - min_val) / range_val
        else:
            df[col + "_norm"] = 0.0
    


    # df["score_avg"] = (df["score_gmm"] + df["score_cadi"]) / 2
    # df["ensemble_score"] = (df["score_gmm"] + df["score_cadi"]) / 2

    alphas = np.linspace(0,1,11)
    
    for alpha in alphas:

        df["ensemble_score"] = (alpha * df["score_gmm_norm"]) +  ((1 - alpha) * df["score_cadi_norm"])
        df["ensemble_qscore"] = (alpha * df["score_gmm_qnorm"]) +  ((1 - alpha) * df["score_cadi_qnorm"])
        scored_path = OUT_DIR / f"alpha_{alpha}_all_scored.csv"
        df.to_csv(scored_path, index=False)
        print(f"Scored features saved to: {scored_path}")

print("COMPLETED PIPELIEN SUCCESFULLY")





