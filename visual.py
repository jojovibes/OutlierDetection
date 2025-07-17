import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

frames_root = Path("/home/jlin1/OutlierDetection/shanghaitech(og)/testing/frames")
masks_root = Path("/home/jlin1/OutlierDetection/shanghaitech(og)/testing/test_pixel_mask")
csv_path = Path("/home/jlin1/OutlierDetection/cleaned_data.csv")


def load_bounding_boxes(csv_path, image_dims):

    df = pd.read_csv(csv_path)
    w, h = image_dims

    df = df[
        (df['x2'] > df['x1']) &
        (df['y2'] > df['y1']) &
        (df['x1'] >= 0) & (df['y1'] >= 0)
    ]

    box_dict = defaultdict(list)
    for _, row in df.iterrows():
        key = f"{row['frame_dir']}/{row['filename']}"
        box = (
            int(row['x1']),
            int(row['y1']),
            int(row['x2']),
            int(row['y2']),
            int(row['class_id'])
        )

        box_dict[key].append(box)

    return box_dict


def visualize_frame_with_mask_and_boxes(image_path, mask_array, frame_idx, bboxes, output_dir):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = mask_array[frame_idx]

    overlay = np.zeros_like(image_rgb)
    overlay[mask > 0] = [255, 0, 0]  
    blended = cv2.addWeighted(image_rgb, 1.0, overlay, 0.4, 0)

    for x1, y1, x2, y2, class_id in bboxes:

        height, width = blended.shape[:2]

        # x1 = max(0, min(x1, width - 1))
        # x2 = max(0, min(x2, width - 1))
        # y1 = max(0, min(y1, height - 1))
        # y2 = max(0, min(y2, height - 1))

        # print(f"Drew box: ({x1}, {y1}, {x2}, {y2})")
        # print(image_path.stem)

        if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
            print(f"Drawing box: ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(blended, (x1, y1), (x2, y2), (225, 255, 0), 2)
            label_text = f"Class {class_id}"
            cv2.putText(
                blended,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )
        # else:
            # print(f"Skipped small/invalid box: ({x1}, {y1}), ({x2}, {y2})")
 
    
    
    # cv2.rectangle(blended, (348, 976), (360, 989), (255, 255, 0), 2)  # yellow test box

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)) 

    plt.figure(figsize=(8, 8))
    plt.imshow(blended)
    plt.axis("off")
    plt.title(f"{image_path.parent.name}/{image_path.name}")
    plt.show()
    plt.close()


def visualize_all_videos():

    box_dict = load_bounding_boxes(csv_path, image_dims=(480, 856))


    for frame_dir_path in sorted(frames_root.iterdir()):
        if not frame_dir_path.is_dir():
            continue

        frame_dir = frame_dir_path.name

        print(f"Processing: {frame_dir}")

        mask_file = masks_root / f"{frame_dir}.npy"
        if not mask_file.exists():
            print(f"No mask found for {frame_dir}, skipping.")
            continue

        try:
            mask_array = np.load(mask_file)
        except Exception as e:
            print(f"Error loading mask for {frame_dir}: {e}")
            continue

        frame_paths = sorted(p for p in frame_dir_path.glob("*.jpg") if not p.name.startswith("._"))

        if len(frame_paths) != len(mask_array):
            print(f"Frame count mismatch: {len(frame_paths)} frames vs {mask_array.shape[0]} masks")
            continue

        for idx, frame_path in enumerate(frame_paths):
            key = f"{frame_dir}/{frame_path.name}"
            bboxes = box_dict.get(key, [])
            output_dir = Path("output_viz") / frame_dir
            visualize_frame_with_mask_and_boxes(frame_path, mask_array, idx, bboxes, output_dir)


def visualize_one_video(frame_dir, box_dict=None):
    print(f"Testing: {frame_dir}")

    frame_dir_path = frames_root / frame_dir
    mask_file = masks_root / f"{frame_dir}.npy"

    if not mask_file.exists():
        print(f"No mask found for {frame_dir}")
        return

    try:
        mask_array = np.load(mask_file)
    except Exception as e:
        print(f"Error loading mask for {frame_dir}: {e}")
        return

    frame_paths = sorted(p for p in frame_dir_path.glob("*.jpg") if not p.name.startswith("._"))

    if len(frame_paths) != len(mask_array):
        print(frame_paths)
        print(mask_array)
        print(f"Frame/mask count mismatch: {len(frame_paths)} vs {mask_array.shape[0]}")
        return

    if box_dict is None:
        box_dict = load_bounding_boxes(csv_path, image_dims=(480, 856))


    for idx, frame_path in enumerate(frame_paths):
        key = f"{frame_dir}/{frame_path.name}"
        bboxes = box_dict.get(key, [])
        output_dir = Path("output_viz") / frame_dir
        visualize_frame_with_mask_and_boxes(frame_path, mask_array, idx, bboxes, output_dir)



if __name__ == "__main__":

    visualize_one_video("12_0151")

    # visualize_all_videos()

