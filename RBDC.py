def compute_rbdc(df, masks_dict, iou_thresh=0.3):
    """
    df: DataFrame with ['frame', 'bbox', 'cadi_anomaly']
    masks_dict: Dict of {frame_idx: binary_mask_array}
    """
    tp, fp, fn = 0, 0, 0

    for frame_idx, group in df.groupby('frame'):
        pred_bboxes = group[group['cadi_anomaly'] == 1]['bbox'].tolist()
        mask = masks_dict.get(frame_idx)
        if mask is None:
            continue

        matched = [False] * len(pred_bboxes)
        mask_used = np.zeros_like(mask, dtype=bool)

        for bbox in pred_bboxes:
            iou = compute_mask_iou(bbox, mask)
            if iou > iou_thresh:
                tp += 1
            else:
                fp += 1

        if mask.sum() > 0:
            fn += 1  # Ground truth anomaly exists but not matched

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1
