from .gdino_process import DetectionResult
import numpy as np
import supervision as sv
from typing import List, Dict
from .ocsort.ocsort import OCSort

def _sv_to_ocsort_array(
    detections: sv.Detections,
    phrases: List[str],
    *,
    min_conf: float,
    include_class: bool,
    class_map: Dict[str, int],
    default_class_id: int
) -> np.ndarray:
    """
    Convert a single frame's (Detections, phrases) to OC-SORT ndarray.
    """
    # Handle empty detections
    if detections is None or len(detections) == 0:
        return np.empty((0, 6 if include_class else 5), dtype=float)

    # Boxes (xyxy already in supervision)
    boxes = detections.xyxy.astype(float)  # (N, 4)

    # Scores (default to 1.0 if missing)
    if detections.confidence is None:
        scores = np.ones((boxes.shape[0],), dtype=float)
    else:
        scores = detections.confidence.astype(float)

    # Confidence filter
    keep = scores >= float(min_conf)
    if not np.any(keep):
        return np.empty((0, 6 if include_class else 5), dtype=float)

    boxes = boxes[keep]
    scores = scores[keep]

    if not include_class:
        return np.hstack([boxes, scores[:, None]])

    # Determine class ids
    if phrases is not None and len(phrases) > 0:
        # Align phrases with kept indices; if lengths mismatch, truncate/pad safely
        N = len(detections)
        phrases_arr = np.array(phrases[:N], dtype=object)
        phrases_kept = phrases_arr[keep]
        if class_map is not None:
            cls = np.array([class_map.get(p, default_class_id) for p in phrases_kept], dtype=int)
        else:
            # If no mapping provided, make per-frame temp ids by hashing (stable within run)
            cls = np.array([abs(hash(p)) % (2**31) for p in phrases_kept], dtype=int)
    else:
        # Fall back to detections.class_id if available
        if detections.class_id is not None:
            cls_all = np.asarray(detections.class_id)
            cls = cls_all[keep].astype(int, copy=False)
        else:
            cls = np.full((boxes.shape[0],), default_class_id, dtype=int)

    return np.hstack([boxes, scores[:, None], cls[:, None]])

## parsing DetectionResult to OC-Sort input compatible format
