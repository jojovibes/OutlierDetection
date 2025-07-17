import sys
import torch
import numpy as np
import gc

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


model = attempt_load(YOLOV7_WEIGHTS, map_location=DEVICE)
# model = attempt_load(YOLOV7_WEIGHTS, map_location=DEVICE, weights_only=False)
model.eval()

tracker = BoTSORT(args,frame_rate=30)

def preprocess_frame(img, img_size=640):
    # img = letterbox(img, (img_size, img_size), stride=32, auto=True)[0]
    img = letterbox(img, (640, 640), auto=False, scaleFill=True)[0]
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

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2Area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

    unionArea = box1Area + box2Area - interArea
    if unionArea == 0:
        return 0.0
    iou = interArea / unionArea
    return iou


def extract_features(img, prev_bboxes, frame_idx=None):
    img_tensor = preprocess_frame(img, IMG_SIZE)

    try:
        with torch.no_grad():
            pred_raw = model(img_tensor, augment=False)[0]
    except Exception as e:
        import traceback
        print("Error in YOLOv7 forward pass:")
        traceback.print_exc()
        return []

    metadata = []

    pred = non_max_suppression(pred_raw, CONF_THRESHOLD, IOU_THRESHOLD)[0]

    if pred is None or not len(pred):
        tracker.update(np.empty((0, 5)), img)
        return []

    pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()

    detections = []
    class_probs_vector = []
    logits_vector = []

    pred_raw_cpu = pred_raw[0].cpu()
    for i, (*xyxy, conf, cls) in enumerate(pred):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        detections.append([x1, y1, x2, y2, conf.cpu().item()])

        # Match this pred box back to a raw prediction using IoU
        best_iou = 0

        best_logits = np.array([0.0] * 80)  # fallback
        best_probs = np.array([0.0] * 80)

        for raw_pred in pred_raw_cpu:
            raw_box = raw_pred[:4].numpy()
            raw_conf = raw_pred[4].item()
            raw_logits = raw_pred[5:].numpy()

            iou = compute_iou([x1, y1, x2, y2], raw_box)
            if iou > best_iou:
                best_iou = iou
                best_logits = raw_logits
                best_probs = (raw_conf * torch.sigmoid(torch.tensor(raw_logits))).numpy()

        logits_vector.append(best_logits)
        class_probs_vector.append(best_probs)

    # for i, (*xyxy, conf, cls) in enumerate(pred):
    #     x1, y1, x2, y2 = [float(v) for v in xyxy]
    #     detections.append([x1, y1, x2, y2, conf.cpu().item()])

    #     cls_logits = pred[i, 5:] if pred.shape[1] > 6 else torch.zeros(80)  # fallback
    #     cls_probs = torch.sigmoid(cls_logits)
    #     probs = (conf * cls_probs).detach().cpu().numpy().tolist()
    #     logits = cls_logits.detach().cpu().numpy().tolist()

    #     class_probs_vector.append(probs)
    #     logits_vector.append(logits)

    detections = np.array(detections)
    outputs = tracker.update(detections, img)

    if outputs is None or len(outputs) == 0:
        print(f"[Frame {frame_idx}] No tracked objects.")
        return []

    RELEVANT_CLASSES = list(range(12))

    for output in outputs:
        x1, y1, x2, y2 = output.tlbr
        track_id = int(output.track_id)
        bbox = [float(x1), float(y1), float(x2), float(y2)]

        max_iou = 0
        matched_class_id = None
        matched_conf = None
        matched_class_probs = None
        matched_logits = None

        for j, det_box in enumerate(pred[:, :4]):
            pred_box = det_box.cpu().numpy()
            iou = compute_iou(bbox, pred_box)
            if iou > max_iou:
                max_iou = iou
                matched_class_id = int(pred[j, 5].item()) if pred.shape[1] > 6 else 0
                matched_conf = float(pred[j, 4].item())
                matched_class_probs = [class_probs_vector[j][k] for k in RELEVANT_CLASSES]
                matched_logits = [logits_vector[j][k] for k in RELEVANT_CLASSES]

        velocity, direction = compute_velocity_direction(bbox, prev_bboxes.get(track_id, bbox))
        prev_bboxes[track_id] = bbox

        metadata.append({
            'class_id': matched_class_id,
            'confidence': matched_conf,
            'track_id': track_id,
            'bbox': bbox,
            'velocity': velocity,
            'direction': direction,
            'class_probabilities': matched_class_probs,
            'logits': matched_logits,
            'max_IOU': max_iou
        })

        print(f"[Frame {frame_idx}] track_id: {track_id}, max_iou: {max_iou}")
        print(f"class_probs : {matched_class_probs}")
        print(f"logits : {matched_logits}")

    torch.cuda.empty_cache()
    gc.collect()

    return metadata
