from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from torchvision.ops import box_convert

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
import supervision as sv
import cv2
from .preprocess import VidTensor
import os
import time

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
        preprocessed_vid_tensor: VidTensor,
        classes: List[str],
        box_threshold: float,
        text_threshold: float,
        save_anotated: bool,
        dir_path: Optional[str]
    ) -> List[Tuple[sv.Detections, List[str]]]:
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
        processed_frames = preprocessed_vid_tensor.vid_tensor.to(torch.device(self.device))
        raw_Res = []
        results = []
        for i in range(int(preprocessed_vid_tensor.total_frame)):
            boxes, logits, phrases = predict(
                model=self.model,
                image=processed_frames[i],
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device)
            raw_Res.append((boxes, logits, phrases))

        ## Saving the file into a directory
        if save_anotated:
            save_to_dir_anotated(preprocessed_vid_tensor, raw_Res, dir_path)
        
        for box, logs, phrases in raw_Res:
            source_h, source_w= int(preprocessed_vid_tensor.height), int(preprocessed_vid_tensor.width)
            detections = Model.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=box,
                logits=logs)
            results.append((detections, phrases))
        return results

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

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
    sequence: List[Tuple[sv.Detections, List[str]]]
) -> Dict[str, int]:
    """
    Build a stable phrase->id mapping across the whole video.
    """
    vocab = []
    seen = set()
    for _, phrases in sequence:
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
    sequence: List[Tuple[sv.Detections, List[str]]],
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
    for dets, phrases in sequence:
        arr = _sv_to_ocsort_array(
            dets,
            phrases,
            min_conf=min_conf,
            include_class=include_class,
            class_map=phrase_id_map,
            default_class_id=default_class_id
        )
        outputs.append(arr)

    return outputs, (phrase_id_map if include_class else None)

def save_to_dir_anotated(vid_tensor:VidTensor, raw_res:List, dirpath):

    if dirpath is None:
        dirpath = "."

    # Ensure directory exists
    os.makedirs(dirpath, exist_ok=True)

    out_name = f"anotated_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = os.path.join(dirpath, out_name)

    vid = cv2.VideoCapture(vid_tensor._file_path)
    if not vid.isOpened():
        return

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    anotate_index = 0
    total_annotations = len(raw_res)
    msec_list = getattr(vid_tensor, "vid_frame_msec_data", [])

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # convert to RGB for annotate (original code used RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_msec = vid.get(cv2.CAP_PROP_POS_MSEC)

        # advance annotation index to match current playback time (safe-guard bounds)
        if msec_list:
            # move forward while current time is greater than target annotation time
            while (anotate_index + 1 < len(msec_list)) and (curr_msec > msec_list[anotate_index]):
                anotate_index += 1

        # pick annotation if available, otherwise keep original frame
        if anotate_index < total_annotations:
            boxes, logits, phrases = raw_res[anotate_index]
            annotated_rgb = annotate(frame_rgb, boxes, logits, phrases)
        else:
            annotated_rgb = frame_rgb

        # lazy-init writer once we know frame size
        if writer is None:
            h, w = int(annotated_rgb.shape[0]), int(annotated_rgb.shape[1])
            writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

        # convert back to BGR for OpenCV writer and write
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        writer.write(annotated_bgr)

    if writer is not None:
        writer.release()
    vid.release()

