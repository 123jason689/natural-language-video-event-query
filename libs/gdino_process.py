import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import annotate, load_model, predict
from torchvision.ops import box_convert

from .typings_ import FrameBatch
from .ocsort.ocsort import OCSort

@dataclass
class DetectionResult:
    frame_index: int
    timestamp_ms: float
    detections: sv.Detections
    phrases: List[str]
    # boxes_cxcywh: torch.Tensor
    # logits: torch.Tensor

class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_classes(
        self,
        frame_batch: FrameBatch,
        classes: List[str],
        box_threshold: float,
        text_threshold: float,
    ) -> List[DetectionResult]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_frames = frame_batch.frames.to(self.device, non_blocking=True)

        results: List[DetectionResult] = []
        source_h = int(frame_batch.height)
        source_w = int(frame_batch.width)

        for i in range(processed_frames.shape[0]):
            boxes, logits, phrases = predict(
                model=self.model,
                image=processed_frames[i],
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device)
            detections = Model.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=boxes,
                logits=logits,
            )
            detections.class_id = Model.phrases2classes(phrases=phrases, classes=classes)

            results.append(
                DetectionResult(
                    frame_index=int(frame_batch.frame_indices[i].item()),
                    timestamp_ms=float(frame_batch.timestamps_ms[i].item()),
                    detections=detections,
                    phrases=phrases,
                    boxes_cxcywh=boxes.cpu(),
                    logits=logits.cpu(),
                )
            )
        return results

    def predict_with_caption(
        self,
        frame_batch: FrameBatch,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        # ocsort_model: OCSort
    ) -> List[DetectionResult]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_caption(
            image=image,
            caption=CAPTION,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        processed_frames = frame_batch.frames.to(self.device, non_blocking=True)

        results: List[DetectionResult] = []
        source_h = int(frame_batch.height)
        source_w = int(frame_batch.width)

        for i in range(processed_frames.shape[0]):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=processed_frames[i],
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device)
            detections = Model.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=boxes,
                logits=logits,
                # ocsort = ocsort_model
            )

            results.append(
                DetectionResult(
                    frame_index=int(frame_batch.frame_indices[i].item()),
                    timestamp_ms=float(frame_batch.timestamps_ms[i].item()),
                    detections=detections,
                    phrases=phrases,
                    # boxes_cxcywh=boxes.cpu(),
                    # logits=logits.cpu(),
                )
            )

        return results
    

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor,
            # ocsort: OCSort
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.cpu().numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)


        # phrase_class_idx = np.arange(xyxy.shape[0])
        # out = np.column_stack([xyxy, confidence, phrase_class_idx])

        # oc_outputs = ocsort.update(out, (source_h, source_w), (source_h, source_w), 3) # dont ask me why it's like this, legacy code babyyyyy.....
        # ## oc sort outputs (x,y,x,y,score, phrase_class_idx, object_id)
        
        # if len(oc_outputs) == 0:
        #     return sv.Detections.empty()

        # tracked_xyxy_pixel = oc_outputs[:, :4]
        # confidence = oc_outputs[:, 4]
        # class_ids = oc_outputs[:, 5].astype(int)
        # tracked_id = oc_outputs[:, 6].astype(int) 

        # tracked_xyxy_norm = tracked_xyxy_pixel / np.array([source_w, source_h, source_w, source_h])

        # return sv.Detections(
        #     xyxy=tracked_xyxy_norm, 
        #     confidence=confidence,
        #     class_id=class_ids,
        #     tracker_id=tracked_id
        # )

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)


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


def build_global_class_map(
    sequence: List[DetectionResult]
) -> Dict[str, int]:
    """
    Build a stable phrase->id mapping across the whole video.
    """
    vocab = []
    seen = set()
    for result in sequence:
        phrases = result.phrases
        if phrases is None:
            continue
        for p in phrases:
            if p is None:
                continue
            if p not in seen:
                seen.add(p)
                vocab.append(p)
    # 0..K-1 ids
    return {p: i for i, p in enumerate(vocab)}


def sequence_to_ocsort_inputs(
    sequence: List[DetectionResult],
    *,
    min_conf: float = 0.25,
    include_class: bool = False,
    use_global_class_map: bool = False,
    class_map: Optional[Dict[str, int]] = None,
    default_class_id: int = -1
) -> Tuple[List[np.ndarray], Optional[Dict[str, int]]]:
    """
    Convert a video sequence of (Detections, phrases) into a list of OC-SORT inputs.

    Parameters
    ----------
    sequence : List[(sv.Detections, List[str])]
        One tuple per frame.
    min_conf : float
        Confidence threshold applied before tracking.
    include_class : bool
        If True, outputs (N, 6): [x1,y1,x2,y2,score,class_id].
        Otherwise (N, 5): [x1,y1,x2,y2,score].
    use_global_class_map : bool
        If True, build a consistent phrase->id map across the *entire* sequence.
        Ignored if include_class=False.
    class_map : Optional[Dict[str,int]]
        Provide your own phrase->id map (overrides auto-building).
    default_class_id : int
        Used when a phrase (or class) is unknown/missing.

    Returns
    -------
    (dets_per_frame, phrase_id_map)
        dets_per_frame: List[np.ndarray] suitable for OC-SORT update per frame.
        phrase_id_map: Dict[str,int] if include_class=True, else None.
    """
    phrase_id_map:Dict[str, int] = dict()
    if include_class:
        if class_map is not None:
            phrase_id_map = class_map
        elif use_global_class_map:
            phrase_id_map = build_global_class_map(sequence)

    outputs: List[np.ndarray] = []
    for result in sequence:
        arr = _sv_to_ocsort_array(
            result.detections,
            result.phrases,
            min_conf=min_conf,
            include_class=include_class,
            class_map=phrase_id_map,
            default_class_id=default_class_id
        )
        outputs.append(arr)

    return outputs, (phrase_id_map if include_class else None)


def save_to_dir_anotated(
    video_path: str,
    frame_results: List[DetectionResult],
    dirpath: Optional[str] = None,
) -> Optional[str]:
    if not frame_results:
        return None

    target_dir = dirpath or "."
    os.makedirs(target_dir, exist_ok=True)

    out_name = f"anotated_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = os.path.join(target_dir, out_name)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video for annotation: {video_path}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None

    result_map = {res.frame_index: res for res in sorted(frame_results, key=lambda r: r.frame_index)}

    last_res = None

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # frame is BGR from OpenCV. keep it BGR.
        annotated_frame = frame
        res = result_map.get(frame_idx)

        if res is not None:
            last_res = res

        if last_res is not None:
            annotated_frame = annotate_bgr(frame, last_res.detections.xyxy, last_res.detections.confidence, last_res.phrases)

        if writer is None:
            h, w = annotated_frame.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

        writer.write(annotated_frame)
        frame_idx += 1

    capture.release()
    if writer is not None:
        writer.release()

    return out_path


# def annotate_bgr(image_bgr: np.ndarray,
#                  boxes: np.ndarray,
#                  logits: np.ndarray,
#                  phrases: List[str],
#                  tracker_ids: np.ndarray) -> np.ndarray:
#     """
#     Draw boxes/labels on a BGR image and return BGR.
#     """
#     # h, w, _ = image_bgr.shape

#     # boxes_abs = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
#     # xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#     detections = sv.Detections(xyxy=boxes)
#     labels = [f"{p} {t} {l:.2f}" for p, l, t in zip(phrases, logits, tracker_ids)]

#     bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
#     label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

#     annotated = image_bgr.copy()
#     annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
#     annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
#     return annotated

def annotate_bgr(image_bgr: np.ndarray,
                 boxes: torch.Tensor,
                 logits: torch.Tensor,
                 phrases: List[str]) -> np.ndarray:
    """
    Draw boxes/labels on a BGR image and return BGR.
    """
    h, w, _ = image_bgr.shape

    # boxes are cx,cy,w,h normalized? you already scale by w,h in your code
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    boxes_abs = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
    xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    detections = sv.Detections(xyxy=xyxy)
    labels = [f"{p} {l:.2f}" for p, l in zip(phrases, logits)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated = image_bgr.copy()
    annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated

def _annotate_bgr(image_bgr: np.ndarray,
                 detections: sv.Detections,
                 phrases: List[str]) -> np.ndarray:
    """
    Draw boxes/labels on a BGR image.
    Expects detections.xyxy to be NORMALIZED (0-1).
    """
    h, w, _ = image_bgr.shape


    scaled_xyxy = detections.xyxy * np.array([w, h, w, h])
    drawing_detections = sv.Detections(
        xyxy=scaled_xyxy,
        class_id=detections.class_id,
        tracker_id=detections.tracker_id
    )

    labels = []
    
    for i in range(len(detections)):
        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
        class_id = detections.class_id[i] if detections.class_id is not None else None
        confidence = detections.confidence[i] if detections.confidence is not None else None
        
        phrase = phrases[class_id] if class_id is not None and class_id < len(phrases) else "unknown"
        label = f"#{tracker_id} {phrase}: {confidence:.2f}"
        labels.append(label)

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated = image_bgr.copy()
    annotated = bbox_annotator.annotate(scene=annotated, detections=drawing_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=drawing_detections, labels=labels)
    
    return annotated

