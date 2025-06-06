import torch
import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from tracker.bot_sort import BoTSORT
import torch.nn.functional as F

from config import YOLOConfig, BoTSORTConfig

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, yolo_config: YOLOConfig, bot_sort_config: BoTSORTConfig):
        self.config = yolo_config
        self.model = self._load_model()
        self.tracker = self._init_tracker(bot_sort_config)
        self.prev_bboxes = {}
        
    def _load_model(self) -> torch.nn.Module:
        """Load and initialize YOLOv7 model."""
        try:
            model = attempt_load(self.config.weights_path, map_location=self.config.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv7 model: {e}")
            raise
            
    def _init_tracker(self, config: BoTSORTConfig) -> BoTSORT:
        """Initialize BoT-SORT tracker."""
        try:
            return BoTSORT(config, frame_rate=30)
        except Exception as e:
            logger.error(f"Failed to initialize BoT-SORT tracker: {e}")
            raise
            
    def preprocess_frame(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLOv7 inference."""
        img = letterbox(img, self.config.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.config.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
        
    def compute_velocity_direction(self, current_bbox: List[float], 
                                 prev_bbox: List[float], fps: int = 30) -> tuple:
        """Compute velocity and direction from bounding boxes."""
        dx = (current_bbox[0] + current_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2
        dy = (current_bbox[1] + current_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
        velocity = np.sqrt(dx**2 + dy**2) * fps
        direction = np.arctan2(dy, dx)
        return velocity, direction
        
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        return interArea / float(box1Area + box2Area - interArea)
        
    def extract_features(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract features from image using YOLOv7 and BoT-SORT."""
        try:
            img_tensor = self.preprocess_frame(img)
            pred_raw = self.model(img_tensor, augment=False)[0]
            
            class_probs_vector = []
            for det in pred_raw:
                if det is None or det.shape[0] == 0:
                    continue
                conf = det[:, 4:5]
                cls_logits = det[:, 5:]
                cls_scores = F.sigmoid(cls_logits)
                class_probs = conf * cls_scores
                class_probs_vector.append(class_probs)
                
            pred = non_max_suppression(pred_raw, self.config.conf_threshold, 
                                     self.config.iou_threshold)[0]
                                     
            if pred is None or not len(pred):
                self.tracker.update(np.empty((0, 5)))
                return []
                
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()
            
            detections = []
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                class_id = int(cls.item())
                detections.append([x1, y1, x2, y2, conf.cpu().item()])
            detections = np.array(detections)
            
            outputs = self.tracker.update(detections, img)
            metadata = []
            
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
                    iou = self.compute_iou(bbox, pred_box)
                    if iou > max_iou:
                        max_iou = iou
                        matched_class_id = int(cls.item())
                        matched_conf = float(conf.item())
                        matched_class_probs = class_probs_vector[0][i].detach().cpu().numpy().tolist()
                        
                velocity, direction = self.compute_velocity_direction(
                    bbox, self.prev_bboxes.get(track_id, bbox))
                    
                metadata.append({
                    'class_id': matched_class_id,
                    'confidence': matched_conf,
                    'track_id': track_id,
                    'bbox': bbox,
                    'velocity': velocity,
                    'direction': direction,
                    'class_probabilities': matched_class_probs
                })
                
                self.prev_bboxes[track_id] = bbox
                
            return metadata
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return [] 