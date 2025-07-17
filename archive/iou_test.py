import numpy as np
# from your_module import compute_mask_iou, compute_true_mask_iou

def compute_mask_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    if np.all(mask[y1:y2, x1:x2]):
        return 1.0

    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.count_nonzero(mask)

    return intersection / union if union > 0 else 0.0


def compute_true_mask_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True


    mask_bool = mask > 0
    intersection = np.logical_and(mask_bool, bbox_mask).sum()
    union = np.logical_or(mask_bool, bbox_mask).sum()

    return intersection / union if union > 0 else 0.0


def test_bbox_inside_mask():
    mask = np.zeros((10, 10), dtype=int)
    mask[2:8, 2:8] = 1  # 6x6 region = 36 pixels
    bbox = (3, 3, 6, 6)  # 3x3 = 9 pixels, fully within

    iou = compute_mask_iou(bbox, mask)

    assert np.isclose(iou, 9 / 36)


def test_bbox_equals_mask():
    mask = np.zeros((10, 10), dtype=int)
    mask[2:8, 2:8] = 1
    bbox = (2, 2, 8, 8)

    iou_asym = compute_mask_iou(bbox, mask)
    iou_true = compute_true_mask_iou(bbox, mask)

    assert np.isclose(iou_asym, 1.0)
    assert np.isclose(iou_true, 1.0)


def test_bbox_outside_mask():
    mask = np.zeros((10, 10), dtype=int)
    mask[2:5, 2:5] = 1
    bbox = (6, 6, 8, 8)

    assert compute_mask_iou(bbox, mask) == 0.0
    assert compute_true_mask_iou(bbox, mask) == 0.0


def test_partial_overlap():
    mask = np.zeros((10, 10), dtype=int)
    mask[2:6, 2:6] = 1  # 4x4 = 16 pixels
    bbox = (4, 4, 8, 8)  # overlaps lower right 2x2 = 4 pixels

    iou_asym = compute_mask_iou(bbox, mask)
    iou_true = compute_true_mask_iou(bbox, mask)

    assert np.isclose(iou_asym, 4 / 16)
    assert np.isclose(iou_true, 4 / (16 + 16 - 4))  # = 4 / 28


def test_zero_union_mask():
    mask = np.zeros((10, 10), dtype=int)
    bbox = (2, 2, 4, 4)

    assert compute_mask_iou(bbox, mask) == 0.0
    assert compute_true_mask_iou(bbox, mask) == 0.0


def test_invalid_bbox():
    mask = np.ones((10, 10), dtype=int)
    bbox = (5, 5, 5, 7)  # zero width

    assert compute_mask_iou(bbox, mask) == 0.0
    assert compute_true_mask_iou(bbox, mask) == 0.0
