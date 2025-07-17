import cv2
from pathlib import Path

def extract_frames_from_videos(video_dir: Path, output_dirname: str = "frames"):
    video_dir = Path(video_dir)
    output_root = video_dir / output_dirname
    output_root.mkdir(exist_ok=True)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    for video_path in video_dir.iterdir():
        if video_path.suffix.lower() not in video_extensions:
            continue

        video_name = video_path.stem
        output_folder = output_root / video_name
        output_folder.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = output_folder / f"{frame_idx:03d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frame_idx += 1

        cap.release()
        print(f"Extracted {frame_idx} frames from {video_path.name} to {output_folder}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("/home/jlin1/OutlierDetection/training")
        sys.exit(1)

    extract_frames_from_videos(Path(sys.argv[1]))
