import sys
import torch
import numpy as np

# sys.path.append('/Users/joelylin/Documents/GitHub/yolov7') 
# sys.path.append('/Users/joelylin/Documents/GitHub/BoT-SORT')

sys.path.append('/home/jlin1/yolov7')
sys.path.append('/home/jlin1/BoT-SORT')


from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from tracker.bot_sort import BoTSORT
import torch.nn.functional as F
from utilz import select_feature_columns


class Args:
    track_high_thresh = 0.6
    track_low_thresh = 0.1
    new_track_thresh = 0.7
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False
    with_reid = False
    proximity_thresh = 0.5
    appearance_thresh = 0.25
    gmc_method = 'sift'         
    use_byte = True             
    fuse_score = False
    size_thresh = 0
    iou_thresh = 0.5
    cmc_method = 'sift'         
    name = 'test'               
    ablation = 'default'        

args = Args()


YOLOV7_WEIGHTS = '/home/jlin1/yolov7/yolov7.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# FRAMES_DIR = '/Volumes/ronni/shanghaitech/testing/frames'
# OUTPUT_METADATA = '/Volumes/ronni/shanghaitech/metadata_output'
# os.makedirs(OUTPUT_METADATA, exist_ok=True)

model = attempt_load(YOLOV7_WEIGHTS, map_location=DEVICE)
# model = attempt_load(YOLOV7_WEIGHTS, map_location=DEVICE, weights_only=False)
model.eval()

tracker = BoTSORT(args,frame_rate=30)

def preprocess_frame(img, img_size=640):
    img = letterbox(img, img_size, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def compute_velocity_direction(current_bbox, prev_bbox, fps=30):
    dx = (current_bbox[0] + current_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2
    dy = (current_bbox[1] + current_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
    velocity = np.sqrt(dx**2 + dy**2) * fps
    direction = np.arctan2(dy, dx)
    return velocity, direction

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


# video_id = "01_0016"
# video_path = os.path.join(FRAMES_DIR, video_id)
# if not os.path.isdir(video_path):
#     raise FileNotFoundError(f"Video directory {video_path} not found!")

def extract_features(img):

    img_tensor = preprocess_frame(img, IMG_SIZE)
    pred_raw = model(img_tensor, augment=False)[0]

    prev_bboxes = {}
    metadata = []

    class_probs_vector = []

    for det in pred_raw:
        if det is None or det.shape[0] == 0:
            continue
        conf = det[:, 4:5]
        cls_logits = det[:, 5:]
        cls_scores = F.sigmoid(cls_logits)  # Apply sigmoid to logits
        class_probs = conf * cls_scores  # YOLO-style: objectness * class_prob
        class_probs_vector.append(class_probs)

    pred = non_max_suppression(pred_raw, CONF_THRESHOLD, IOU_THRESHOLD)[0]

    if pred is None or not len(pred):
        tracker.update(np.empty((0, 5)))

    pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()

    detections = []
    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        class_id = int(cls.item())
        detections.append([x1, y1, x2, y2, conf.cpu().item()])
    detections = np.array(detections)

    outputs = tracker.update(detections, img)

    for output in outputs:
        x1, y1, x2, y2 = output.tlbr
        track_id = int(output.track_id)
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        matched_class_id = None
        matched_conf = None
        matched_class_probs = None
        max_iou = 0

        for i, (*xyxy, conf, cls) in enumerate(pred):
            pred_box = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
            iou = compute_iou(bbox, pred_box)
            if iou > max_iou:
                max_iou = iou
                matched_class_id = int(cls.item())
                matched_conf = float(conf.item())
                matched_class_probs = class_probs_vector[0][i].detach().cpu().numpy().tolist() 

        velocity, direction = compute_velocity_direction(bbox, prev_bboxes.get(track_id, bbox))

        metadata.append({
            'class_id': matched_class_id,
            'confidence': matched_conf,
            'track_id': track_id,
            'bbox': bbox,
            'velocity': velocity,
            'direction': direction,
            'class_probabilities': matched_class_probs 
        })

    prev_bboxes[track_id] = bbox

    return metadata



    # print(f"Frame: {frame_idx}, Class ID: {matched_class_id}, Confidence: {matched_conf:.2f}, Track ID: {track_id}, Velocity: {velocity:.2f}, Direction: {direction:.2f}")

    # output_file = os.path.join(OUTPUT_METADATA, f"{video_id}_metadata.json")
    # with open(output_file, 'w') as f:
    #     json.dump(metadata, f, indent=2)
    # print(f"\n Saved metadata for {video_id} to {output_file}")
   