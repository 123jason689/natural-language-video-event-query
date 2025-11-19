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
        ocsort_model: OCSort
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
                ocsort = ocsort_model
            )

            # if len(boxes) > 0:
            #     print("\n"+">>"*20 + f"\nHere is the sample output with shape {boxes.shape}\n")
            #     print(boxes[0])
            #     print()
            #     print(f"confidence: {logits[0]}")
            #     print("\n"+">>"*20)
            # else:
            #     print("\n"+">>"*20 + "\nNo output / zero detected\n\n")
            #     print("\n"+">>"*20)


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
            ocsort: OCSort
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        phrase_class_idx = np.arange(xyxy.shape[0])
        out = np.column_stack([xyxy, confidence, phrase_class_idx])

        # if len(out) > 0:
        #     print("\n"+">>"*20 + f"\nHere is the sample output with shape {out.shape}\n")
        #     print(out[0])
        #     print("\n"+">>"*20)
        # else:
        #     print("\n"+">>"*20 + "\nNo output / zero detected\n")
        #     print(xyxy.shape)
        #     print(confidence.shape)
        #     print(phrase_class_idx.shape)
        #     print(out.shape)
        #     print("\n"+">>"*20)

        oc_outputs = ocsort.update(out, (source_h, source_w), (source_h, source_w)) # dont ask me why it's like this, legacy code babyyyyy.....
        ## oc sort outputs (x,y,x,y,score, phrase_class_idx, object_id)

        
        return sv.Detections(xyxy=oc_outputs[:, : 4], confidence=oc_outputs[:, 4], class_id=oc_outputs[:, 5], tracker_id=oc_outputs[:, 6])

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

    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # frame is BGR from OpenCV. keep it BGR.
        annotated_frame = frame
        res = result_map.get(frame_idx)
        if res is not None:
            ## i need to translate the class ids from the detections.class_id into the corresponding phrases
            phrases = res.phrases
            p = [phrases[idx] for idx in res.detections.class_id.astype(np.int32)]
            annotated_frame = annotate_bgr(frame, res.detections.xyxy, res.detections.confidence, p, res.detections.tracker_id.astype(np.int32))

        if writer is None:
            h, w = annotated_frame.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

        writer.write(annotated_frame)
        frame_idx += 1

    capture.release()
    if writer is not None:
        writer.release()

    return out_path


def annotate_bgr(image_bgr: np.ndarray,
                 boxes: np.ndarray,
                 logits: np.ndarray,
                 phrases: List[str],
                 tracker_ids: np.ndarray) -> np.ndarray:
    """
    Draw boxes/labels on a BGR image and return BGR.
    """
    # h, w, _ = image_bgr.shape

    # boxes_abs = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
    # xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    detections = sv.Detections(xyxy=boxes)
    labels = [f"{p} {t} {l:.2f}" for p, l, t in zip(phrases, logits, tracker_ids)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated = image_bgr.copy()
    annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated

