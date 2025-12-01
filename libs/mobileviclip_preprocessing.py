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
    
    Key Features:
    - Temporal Resampling: Selects 8 frames evenly distributed over a fixed duration (e.g., 3s)
      regardless of the input detection FPS.
    - Context-aware Cropping: Focuses on the person/action.
    """

    def __init__(
        self,
        target_size: int = 224,
        context_padding: float = 0.2,
        clip_length: int = 8,        # Model requirement (MobileViCLIP uses 8 frames)
        clip_duration: float = 3.0,  # Desired physical duration of the clip in seconds
        clip_stride: int = 4,        # Sliding window stride (in processed frames)
        person_keywords: List[str] = None
    ):
        """
        Args:
            target_size: The H/W resolution for MobileViCLIP (default 224).
            context_padding: Percentage to expand the box to capture context (default 20%).
            clip_length: Number of frames the model expects (T). Fixed at 8 for MobileViCLIP.
            clip_duration: The temporal span (in seconds) the 8 frames should cover.
                           MobileViCLIP prefers 2-10s. Default 3.0s.
            clip_stride: Step size for sliding window (how many frames to move forward).
            person_keywords: List of words to identify a person class.
        """
        self.target_size = target_size
        self.context_padding = context_padding
        self.clip_length = clip_length
        self.clip_duration = clip_duration
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
        # Structure: buffer[track_id] = list of (timestamp_seconds, crop_tensor)
        track_buffers = {tid: [] for tid in selected_track_ids}
        
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
                # Pass timestamp explicitly
                ts_sec = result.timestamp_ms / 1000.0
                self._process_frame(frame_bgr, result, selected_track_ids, track_buffers, ts_sec)
            
            frame_idx += 1
        
        cap.release()

        # 4. Temporal Windowing (Create Clips with Resampling)
        final_output = {}
        for tid, buffer_data in track_buffers.items():
            clips = self._create_temporal_clips(buffer_data)
            if clips:
                final_output[tid] = clips
                print(f"Track ID {tid}: Generated {len(clips)} clips spanning {self.clip_duration}s each.")
        
        return final_output

    def _analyze_tracks(self, results: List[DetectionResult], object_map: ObjectMap) -> Dict:
        # ... (Same as before) ...
        tracks = defaultdict(lambda: {'scores': [], 'class_ids': set()})

        for res in results:
            if res.detections is None: 
                continue
            for i, tracker_id in enumerate(res.detections.tracker_id):
                if tracker_id is None: continue
                tid = int(tracker_id)
                score = float(res.detections.confidence[i])
                cls_id = int(res.detections.class_id[i])
                tracks[tid]['scores'].append(score)
                tracks[tid]['class_ids'].add(cls_id)

        processed_tracks = {}
        for tid, data in tracks.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            is_person = False
            for cid in data['class_ids']:
                class_phrase = object_map.object_map.inverse.get(cid, "")
                if any(k in class_phrase.lower() for k in self.person_keywords):
                    is_person = True
                    break
            if is_person:
                processed_tracks[tid] = avg_score

        return processed_tracks

    def _select_top_k_tracks(self, track_metadata: Dict[int, float], k: int) -> List[int]:
        sorted_tracks = sorted(track_metadata.items(), key=lambda item: item[1], reverse=True)
        return [t[0] for t in sorted_tracks[:k]]

    def _process_frame(
        self, 
        full_frame: np.ndarray, 
        result: DetectionResult, 
        target_ids: List[int], 
        buffers: Dict[int, List],
        timestamp: float
    ):
        """
        Extracts crops and stores them with timestamps.
        """
        if result.detections is None: return

        img_h, img_w = full_frame.shape[:2]
        frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)

        for i, tracker_id in enumerate(result.detections.tracker_id):
            if tracker_id is None: continue
            tid = int(tracker_id)
            
            if tid in target_ids:
                box_norm = result.detections.xyxy[i]
                crop_tensor = self._extract_square_crop(frame_rgb, box_norm, img_w, img_h)
                # Store Tuple: (timestamp, tensor)
                buffers[tid].append((timestamp, crop_tensor))

    def _extract_square_crop(self, frame: np.ndarray, box_norm: np.ndarray, img_w: int, img_h: int) -> torch.Tensor:
        # ... (Same cropping logic as before) ...
        x1, y1, x2, y2 = box_norm * np.array([img_w, img_h, img_w, img_h])
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        max_side = max(w, h)
        square_size = int(max_side * (1 + self.context_padding))
        crop_x1 = int(cx - square_size / 2)
        crop_y1 = int(cy - square_size / 2)
        crop_x2 = crop_x1 + square_size
        crop_y2 = crop_y1 + square_size
        
        pad_l = max(0, -crop_x1)
        pad_t = max(0, -crop_y1)
        pad_r = max(0, crop_x2 - img_w)
        pad_b = max(0, crop_y2 - img_h)
        
        if any([pad_l, pad_t, pad_r, pad_b]):
            frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crop_x1 += pad_l
            crop_x2 += pad_l
            crop_y1 += pad_t
            crop_y2 += pad_t
            
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            crop = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return tensor

    def _create_temporal_clips(self, buffer_data: List[Tuple[float, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Resamples the buffer to create fixed-size clips (8 frames) that span 
        approximately `self.clip_duration`.
        
        Args:
            buffer_data: List of (timestamp, tensor) tuples sorted by time.
        """
        if len(buffer_data) < self.clip_length:
            return []

        clips = []
        
        # buffer_data is a list of frames for a SINGLE track.
        # We iterate through valid start points.
        
        # Pre-calculate timestamps array for faster lookup
        timestamps = np.array([x[0] for x in buffer_data])
        
        # Iterate over the buffer
        for start_idx in range(0, len(buffer_data), self.clip_stride):
            start_time = timestamps[start_idx]
            target_end_time = start_time + self.clip_duration
            
            # Find the index in the buffer closest to target_end_time
            # We search only after start_idx to ensure forward progress
            # searchsorted returns the index where target_end_time should be inserted
            end_search = np.searchsorted(timestamps, target_end_time)
            
            # Adjust end_search to be a valid index
            end_idx = min(end_search, len(buffer_data) - 1)
            
            # Safety check: Is the actual duration reasonably close to target?
            # e.g., if track ends abruptly, we might not have enough footage.
            actual_duration = timestamps[end_idx] - start_time
            
            # If the available window is too short (e.g., < 50% of target), skip
            if actual_duration < (self.clip_duration * 0.5):
                continue

            # Define the window of indices available
            window_indices = np.arange(start_idx, end_idx + 1)
            
            if len(window_indices) < self.clip_length:
                # Not enough frames even to fill the required 8 frames
                # We can either skip or duplicate. Skipping is safer for retrieval precision.
                continue

            # Uniformly sample 8 indices from this window
            # linspace generates evenly spaced numbers
            sampled_indices = np.linspace(
                start_idx, 
                end_idx, 
                self.clip_length, 
                dtype=int
            )
            
            # Gather the tensors
            clip_frames = [buffer_data[i][1] for i in sampled_indices]
            
            # Stack: list of (C, H, W) -> (T, C, H, W)
            clip_tensor = torch.stack(clip_frames, dim=0)
            
            clips.append(clip_tensor)
            
        return clips