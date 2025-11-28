import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, deque

# Import types from your existing codebase
from .gdino_process import DetectionResult
from .typings_ import ObjectMap

class MobileViClipPreprocessor:
    """
    Preprocessing pipeline to convert GDINO+OC-SORT DetectionResults into 
    MobileViCLIP-ready tensor clips.
    
    Order of Operations:
    1. Filter tracks by class (Person/Human only).
    2. Rank tracks by confidence and select Top-K.
    3. Iterate video: Smooth coordinates -> Square Crop (w/ padding) -> Resize.
    4. Temporal Windowing: Group frames into fixed-size clips.
    """

    def __init__(
        self,
        target_size: int = 224,
        context_padding: float = 0.2,
        clip_length: int = 8,
        clip_stride: int = 4,
        person_keywords: List[str] = None
    ):
        """
        Args:
            target_size: The H/W resolution for MobileViCLIP (default 224).
            context_padding: Percentage to expand the box to capture context (default 20%).
            clip_length: Number of frames per clip (T).
            clip_stride: Step size for sliding window (overlap).
            person_keywords: List of words to identify a person class. 
                             Defaults to ["person", "human", "man", "woman", "child", "someone"].
        """
        self.target_size = target_size
        self.context_padding = context_padding
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.person_keywords = person_keywords or ["person", "human", "man", "woman", "child", "someone"]

    def run(
        self, 
        video_path: str, 
        detection_results: List[DetectionResult], 
        object_map: ObjectMap,
        top_k: int = 1
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Main entry point. Returns a dictionary mapping Tracker ID -> List of Clip Tensors.
        
        Returns:
            Dict[int, List[torch.Tensor]]: 
                Key: Tracker ID
                Value: List of tensors, each shape (C, T, H, W) normalized [0-1].
        """
        # 1. Index and Filter Tracks
        print(f"Indexing {len(detection_results)} frames of detection results...")
        track_metadata = self._analyze_tracks(detection_results, object_map)
        
        # 2. Select Top-K Persons
        selected_track_ids = self._select_top_k_tracks(track_metadata, top_k)
        if not selected_track_ids:
            print("No valid person tracks found to process.")
            return {}
            
        print(f"Processing Top-{len(selected_track_ids)} Tracks: {selected_track_ids}")

        # 3. Process Video and Extract Crops
        # Structure: buffer[track_id] = list of resized tensors
        track_buffers = {tid: [] for tid in selected_track_ids}
        
        # We need to map frame_index -> detections for random access logic during stream
        frame_map = {res.frame_index: res for res in detection_results}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_map:
                result = frame_map[frame_idx]
                self._process_frame(frame_bgr, result, selected_track_ids, track_buffers)
            
            frame_idx += 1
        
        cap.release()

        # 4. Temporal Windowing (Create Clips)
        final_output = {}
        for tid, frames_list in track_buffers.items():
            clips = self._create_temporal_clips(frames_list)
            if clips:
                final_output[tid] = clips
                print(f"Track ID {tid}: Generated {len(clips)} clips.")
        
        return final_output

    def _analyze_tracks(self, results: List[DetectionResult], object_map: ObjectMap) -> Dict:
        """
        Aggregates stats for every track to determine class and average score.
        """
        # Dict structure: { track_id: { 'scores': [], 'class_ids': set() } }
        tracks = defaultdict(lambda: {'scores': [], 'class_ids': set()})

        for res in results:
            if res.detections is None: 
                continue
                
            # Iterate through detections in this frame
            for i, tracker_id in enumerate(res.detections.tracker_id):
                if tracker_id is None: continue
                
                tid = int(tracker_id)
                score = float(res.detections.confidence[i])
                cls_id = int(res.detections.class_id[i])
                
                tracks[tid]['scores'].append(score)
                tracks[tid]['class_ids'].add(cls_id)

        # Resolve class names
        processed_tracks = {}
        for tid, data in tracks.items():
            # Calculate metrics
            avg_score = sum(data['scores']) / len(data['scores'])
            
            # Determine if this track is a person
            is_person = False
            for cid in data['class_ids']:
                # object_map.object_map is a bidict. inverse gets the string key from int id
                class_phrase = object_map.object_map.inverse.get(cid, "")
                if any(k in class_phrase.lower() for k in self.person_keywords):
                    is_person = True
                    break
            
            if is_person:
                processed_tracks[tid] = avg_score

        return processed_tracks

    def _select_top_k_tracks(self, track_metadata: Dict[int, float], k: int) -> List[int]:
        """
        Sorts tracks by average confidence score and returns top K IDs.
        """
        # Sort by score descending
        sorted_tracks = sorted(track_metadata.items(), key=lambda item: item[1], reverse=True)
        return [t[0] for t in sorted_tracks[:k]]

    def _process_frame(
        self, 
        full_frame: np.ndarray, 
        result: DetectionResult, 
        target_ids: List[int], 
        buffers: Dict[int, List]
    ):
        """
        Extracts crops for the specific frame for target tracks.
        """
        if result.detections is None: return

        # Get image dims for denormalization
        img_h, img_w = full_frame.shape[:2]
        
        frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)

        for i, tracker_id in enumerate(result.detections.tracker_id):
            if tracker_id is None: continue
            tid = int(tracker_id)
            
            if tid in target_ids:
                # Get normalized box [x1, y1, x2, y2]
                box_norm = result.detections.xyxy[i]
                
                # Crop and Resize
                crop_tensor = self._extract_square_crop(frame_rgb, box_norm, img_w, img_h)
                buffers[tid].append(crop_tensor)

    def _extract_square_crop(
        self, 
        frame: np.ndarray, 
        box_norm: np.ndarray, 
        img_w: int, 
        img_h: int
    ) -> torch.Tensor:
        """
        Logic:
        1. De-normalize coordinates.
        2. Calculate Center.
        3. Determine Square Size (max side * padding).
        4. Crop & Padding (if out of bounds).
        5. Resize to target_size.
        """
        x1, y1, x2, y2 = box_norm * np.array([img_w, img_h, img_w, img_h])
        
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        
        # Square size with context padding
        max_side = max(w, h)
        square_size = int(max_side * (1 + self.context_padding))
        
        # Calculate crop coordinates
        crop_x1 = int(cx - square_size / 2)
        crop_y1 = int(cy - square_size / 2)
        crop_x2 = crop_x1 + square_size
        crop_y2 = crop_y1 + square_size
        
        # Handle Out-of-Bounds by clamping or padding
        # Simple clamping approach (might distort slightly at edges, but robust)
        # For strict MobileViCLIP, we usually want black padding if context is missing,
        # but clamping is standard for simple pipelines.
        
        # Pad image if crop goes outside
        pad_l = max(0, -crop_x1)
        pad_t = max(0, -crop_y1)
        pad_r = max(0, crop_x2 - img_w)
        pad_b = max(0, crop_y2 - img_h)
        
        if any([pad_l, pad_t, pad_r, pad_b]):
            frame = cv2.copyMakeBorder(
                frame, pad_t, pad_b, pad_l, pad_r, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # Adjust coordinates due to padding
            crop_x1 += pad_l
            crop_x2 += pad_l
            crop_y1 += pad_t
            crop_y2 += pad_t
            
        # Crop
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize
        if crop.size == 0:
            # Fallback for degenerate boxes
            crop = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
        # To Tensor (C, H, W) and normalize to [0, 1]
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return tensor

    def _create_temporal_clips(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies windowing to create clips of shape (C, T, H, W).
        MobileViCLIP usually expects (B, C, T, H, W).
        """
        if len(frames) < self.clip_length:
            return []

        clips = []
        # Sliding window
        for i in range(0, len(frames) - self.clip_length + 1, self.clip_stride):
            window = frames[i : i + self.clip_length]
            
            # Stack: list of (C, H, W) -> (T, C, H, W)
            clip_tensor = torch.stack(window, dim=0)
            
            # Permute to (C, T, H, W) as per standard 3D CNN / ViT video inputs
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)
            
            clips.append(clip_tensor)
            
        return clips