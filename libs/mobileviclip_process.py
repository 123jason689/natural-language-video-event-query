import torch
import torch.nn.functional as F
from mobileviclip.models.mobileviclip_small import MobileViCLIP_Small
from mobileviclip.utils.config import Config
from libs.text_prompt import kinetics_templates
from typing import Dict, List

class Model:
    def __init__(self, config_path, base_ckpt_path, finetuned_ckpt_path, device="cuda"):
        self.device = device
        
        # 1. Load Config
        cfg = Config.from_file(config_path)
        
        # Critical settings to avoid crashes
        cfg.model.vision_encoder.use_flash_attn = False 
        cfg.model.vision_encoder.use_fused_rmsnorm = False
        cfg.model.vision_encoder.use_fused_mlp = False
        
        # 2. Set Base Weights (mobileclip_s2.pt) for Initialization
        # The model class needs these to build the backbone correctly
        cfg.model.vision_ckpt_path = base_ckpt_path
        cfg.model.text_ckpt_path = base_ckpt_path
        
        print(f"Initializing MobileViCLIP with base weights: {base_ckpt_path}")
        # is_pretrain=False because we are doing inference
        self.model = MobileViCLIP_Small(cfg, is_pretrain=False)
        
        # 3. Load Fine-Tuned Weights (mobileviclip_small.pt)
        # This overwrites the base weights with the smart, trained weights
        print(f"Loading fine-tuned weights: {finetuned_ckpt_path}")
        checkpoint = torch.load(finetuned_ckpt_path, map_location="cpu")
        
        # Handle different checkpoint formats (sometimes wrapped in 'model' or 'module')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
            
        # Load into model (strict=False allows ignoring minor mismatches if any)
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Status: {msg}")
        
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def encode_text(self, prompt: str):
        # Use template to match training distribution
        text_input = kinetics_templates[2].format(prompt)
        tokens = self.tokenizer(text_input).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_video_clips(self, clips: torch.Tensor):
        # clips shape: [Batch, Channels, Frames, Height, Width]
        clips = clips.to(self.device)
        with torch.no_grad():
            video_features = self.model.encode_vision(clips, test=True)
            video_features = F.normalize(video_features, dim=-1)
        return video_features


    def find_event(self, track_clips: Dict[int, List[torch.Tensor]], prompt:str, clip_stride:int, target_fps:int, clip_duration:float):
        results_timeline = []

        print(f"Searching for event: '{prompt}'")
        text_embeddings = self.encode_text(prompt)

        for track_id, clips in track_clips.items():
            if not clips: continue

            print(f"Track {track_id}: Analyzing {len(clips)} segments...")

            # Stack clips: [Batch, C, T, H, W]
            # Note: If you have too many clips, you might need a mini-batch loop here to avoid OOM            
            batch_clips = torch.stack(clips).to(self.device)

            # Encode Video Clips
            video_embs = self.encode_video_clips(batch_clips)

            # Compute Similarity
            # [Batch, Dim] @ [Dim, 1] -> [Batch]
            scores = (video_embs @ text_embeddings.T).squeeze()
            if scores.ndim == 0: scores = scores.unsqueeze(0)
            
            scores_np = scores.cpu().numpy()
            
            # Map back to approximate timestamps
            # We know each clip represents 'CLIP_DURATION' seconds
            # And we step by 'clip_stride' (processed frames)
            # processed_fps was TARGET_FPS (6)
            stride_sec = clip_stride / target_fps

            for idx, score in enumerate(scores_np):
                start_time = idx * stride_sec
                end_time = start_time + clip_duration
                
                # Threshold filtering (0.25 is a common starting point for CLIP-based models)
                if score > 0.25:
                    results_timeline.append({
                        "track_id": track_id,
                        "start": start_time,
                        "end": end_time,
                        "score": float(score)
                    })
        
        return results_timeline
    
    def process_timeline(results_timeline, overlap_threshold=0.5):
        """
        Merges overlapping or adjacent clips for the same track ID into single event segments.
        
        Args:
            results_timeline: List of dicts [{'track_id': 101, 'start': 0.0, 'end': 3.0, 'score': 0.85}, ...]
            overlap_threshold: Not used directly in simple merging, but useful concept. 
                            Here we merge if clips overlap or are adjacent.
        
        Returns:
            List of merged events: [{'track_id': 101, 'start': 0.0, 'end': 6.0, 'score': 0.88}, ...]
        """
        if not results_timeline:
            return []

        # 1. Group by Track ID
        events_by_track = {}
        for res in results_timeline:
            tid = res['track_id']
            if tid not in events_by_track:
                events_by_track[tid] = []
            events_by_track[tid].append(res)

        merged_events = []

        # 2. Process each track independently
        for tid, clips in events_by_track.items():
            # Sort by start time is CRITICAL for simple merging logic
            clips.sort(key=lambda x: x['start'])
            
            current_event = None
            
            for clip in clips:
                if current_event is None:
                    # Start a new event
                    current_event = {
                        'track_id': tid,
                        'start': clip['start'],
                        'end': clip['end'],
                        'scores': [clip['score']] # Keep list to avg later
                    }
                else:
                    # Check for overlap or adjacency
                    # Logic: If the new clip starts within 'overlap_threshold' seconds of the current event ending
                    if clip['start'] <= current_event['end'] + overlap_threshold:
                        # Merge!
                        current_event['end'] = max(current_event['end'], clip['end'])
                        current_event['scores'].append(clip['score'])
                    else:
                        # Gap found -> Finalize current event and start new one
                        # Calculate average score
                        current_event['score'] = sum(current_event['scores']) / len(current_event['scores'])
                        del current_event['scores'] # Cleanup
                        merged_events.append(current_event)
                        
                        # Start new
                        current_event = {
                            'track_id': tid,
                            'start': clip['start'],
                            'end': clip['end'],
                            'scores': [clip['score']]
                        }
            
            # Don't forget the last event in the loop
            if current_event:
                current_event['score'] = sum(current_event['scores']) / len(current_event['scores'])
                del current_event['scores']
                merged_events.append(current_event)

        # 3. Sort final results by score (highest confidence first)
        merged_events.sort(key=lambda x: x['score'], reverse=True)
        
        return merged_events

