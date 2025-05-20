def compute_mask_iou(bbox, mask):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2+1, x1:x2+1] = True

    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.logical_or(mask, bbox_mask).sum()

    return intersection / union if union > 0 else 0.0
