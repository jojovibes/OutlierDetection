import os
import cv2

src_root = "/home/jlin1/OutlierDetection/UCSDped2/Test"
dst_root = "/home/jlin1/OutlierDetection/UCSDped2/frames"  # where pipeline_runner expects folders
os.makedirs(dst_root, exist_ok=True)

for folder in sorted(os.listdir(src_root)):
    if not folder.startswith("Test") or folder.endswith("_gt"):
        continue

    src_folder = os.path.join(src_root, folder)
    dst_folder = os.path.join(dst_root, folder)
    os.makedirs(dst_folder, exist_ok=True)

    for tif_file in sorted(os.listdir(src_folder)):
        if not tif_file.endswith(".tif"):
            continue
        img = cv2.imread(os.path.join(src_folder, tif_file))
        if img is None:
            continue
        frame_idx = int(os.path.splitext(tif_file)[0])
        out_name = f"{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(dst_folder, out_name), img)

    print(f"Converted {folder} to jpg")
