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
from bidict import bidict

from .typings_ import FrameBatch, ObjectMap
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
        ocsort_model: OCSort,
        object_map:ObjectMap
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
                
            phrases_idx = list()
            for p in phrases:
                isthere = object_map.object_map.get(p)
                if isthere == None:
                    object_map.last_idx += 1
                    object_map.object_map[p] = object_map.last_idx
                    phrases_idx.append(object_map.last_idx)
                else :
                    phrases_idx.append(isthere)

            detections = Model.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=boxes,
                logits=logits,
                ocsort = ocsort_model,
                phrase_class_idx = phrases_idx
            )

            results.append(
                DetectionResult(
                    frame_index=int(frame_batch.frame_indices[i].item()),
                    timestamp_ms=float(frame_batch.timestamps_ms[i].item()),
                    detections=detections,
                    phrases=phrases,
                )
            )

        return results
    

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor,
            ocsort: OCSort,
            phrase_class_idx:List
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.cpu().numpy()

        assert xyxy.shape[0] == len(phrase_class_idx) == confidence.shape[0], "boxes amount not same as the classes length or logits amount not the same amount as xyxy"

        out = np.column_stack([xyxy, confidence, phrase_class_idx])

        oc_outputs = ocsort.update(out, (source_h, source_w), (source_h, source_w), 3) # dont ask me why it's like this, legacy code babyyyyy.....
        ## oc sort outputs (x,y,x,y,score, phrase_class_idx, object_id)

        tracked_xyxy_pixel = oc_outputs[:, :4]
        confidence = oc_outputs[:, 4]
        class_ids = oc_outputs[:, 5].astype(int)
        tracked_id = oc_outputs[:, 6].astype(int) 

        tracked_xyxy_norm = tracked_xyxy_pixel / np.array([source_w, source_h, source_w, source_h])

        print(f"Length of tracked objects = {tracked_id.shape[0]}")

        assert tracked_xyxy_norm.shape[0] == class_ids.shape[0] == confidence.shape[0] == tracked_id.shape[0], "OC Sort output wrong shape or not consistent or NoneType"
        
        return sv.Detections(
            xyxy=tracked_xyxy_norm, 
            confidence=confidence,
            class_id=class_ids,
            tracker_id=tracked_id
        )

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


def save_to_dir_anotated(
    video_path: str,
    frame_results: List[DetectionResult],
    object_map:ObjectMap,
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
            annotated_frame = annotate_bgr(frame, last_res.detections, object_map)

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
                 detections: sv.Detections,
                 object_map: ObjectMap) -> np.ndarray:
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

    labels = [
        f"#{tracker_id} {object_map.object_map.inverse.get(class_id)}: {confidences:.2f}" 
        for tracker_id, class_id, confidences 
        in zip(detections.tracker_id, detections.class_id, detections.confidence)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated = image_bgr.copy()
    annotated = bbox_annotator.annotate(scene=annotated, detections=drawing_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=drawing_detections, labels=labels)
    
    return annotated

def unnormbbox(bbox, w, h):
    return bbox * np.array([w, h, w, h])