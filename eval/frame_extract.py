import cv2
import os

video_dir = "/home/jlin1/OutlierDetection/Avenue/videos"
output_root = "/home/jlin1/OutlierDetection/Avenue/frames"

for video_file in os.listdir(video_dir):
    if not video_file.endswith(".avi"):
        continue

    name = os.path.splitext(video_file)[0]
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{idx:06d}.jpg"
        cv2.imwrite(os.path.join(out_dir, frame_name), frame)
        idx += 1

    cap.release()
    print(f"Extracted frames for {name}")
