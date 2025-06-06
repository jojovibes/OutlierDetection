import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class YOLOConfig:
    weights_path: str = '/Users/joelylin/Documents/GitHub/yolov7/yolov7.pt'
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class BoTSORTConfig:
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    min_box_area: int = 10
    mot20: bool = False
    with_reid: bool = False
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    gmc_method: str = 'sift'
    use_byte: bool = True
    fuse_score: bool = False
    size_thresh: float = 0
    iou_thresh: float = 0.5
    cmc_method: str = 'sift'
    name: str = 'test'
    ablation: str = 'default'

@dataclass
class CADIConfig:
    nb_trees: int = 100
    max_height: int = 256
    contamination_rate: float = 0.05

@dataclass
class PipelineConfig:
    root_dir: str = '/Volumes/ronni/shanghaitech/testing/small_batch'
    output_dir: str = os.path.join(root_dir, "output")
    yolo: YOLOConfig = YOLOConfig()
    bot_sort: BoTSORTConfig = BoTSORTConfig()
    cadi: CADIConfig = CADIConfig()
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "scores"), exist_ok=True) 