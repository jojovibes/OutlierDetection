import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2

from preprocess import extract_features
from GMM import run as run_GMM
from IF import run as run_IF
from utilz import derive_features
from outlierDetection import run as run_cadi

ROOT_DIR = '/Volumes/ronni/shanghaitech/testing/frames'
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_folder(folder_path):
    # print(f"Processing folder: {folder_path}")
    data = []

    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        # print(f"Processing file: {fpath}")
        if fname.startswith("._"): 
            continue
        try:
            frame_idx = int(os.path.splitext(fname)[0])

            if frame_idx == 30: #small_batch testing
                break

            img = cv2.imread(fpath)
            if img is None:
                print(f"Could not read image {fpath}. Skipping.")
                continue

            features_list = extract_features(img)

            if not features_list:
                print(f"No features found in {fpath}. Skipping.")
                continue

            for features in features_list:
                features['filename'] = fname
                features['frame_idx'] = frame_idx
                data.append(features)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return pd.DataFrame(data)


def main():
    for folder_name in os.listdir(ROOT_DIR):
        if folder_name == "output":
            continue

        folder_path = os.path.join(ROOT_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        df = process_folder(folder_path)
        df = derive_features(df)
        
        class_prob_df = pd.DataFrame(df['class_probabilities'].tolist(), index=df.index)
        class_prob_df.columns = [f'class_prob_{i}' for i in range(class_prob_df.shape[1])]
        df = pd.concat([df, class_prob_df], axis=1)
        df.drop(columns=['class_probabilities'], inplace=True)
        df.drop(columns=['bbox'], inplace=True)
        df.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)

        exclude_cols = ['track_id', 'filename', 'frame_idx']
        scale_cols = [col for col in df.columns if col not in exclude_cols]

        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        csv_path = os.path.join(OUTPUT_DIR, f"{folder_name}_features.csv")
        df.to_csv(csv_path, index=False)

        df['score_if'] = run_IF(df)
        df['score_gmm'] = run_GMM(df)
        df = run_cadi(df)
        df['score_avg'] = (df['score_gmm'] + df['score_cadi']) / 2

        out_path = os.path.join(OUTPUT_DIR, f"{folder_name}_scored.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved scored CSV to {out_path}")

if __name__ == "__main__":
    main()
