import torch
import torch.nn.functional as F
import numpy as np
from mobileviclip.models.mobileviclip_small import MobileViCLIP_Small
from mobileviclip.utils.config import Config
from libs.text_prompt import kinetics_templates
from typing import Dict, List, Union

class Model:
    def __init__(self, config_path, base_ckpt_path, finetuned_ckpt_path, device="cuda"):
        self.device = device
        
        # 1. Load Config
        cfg = Config.from_file(config_path)
        
        # Critical settings to avoid crashes
        cfg.model.vision_encoder.use_flash_attn = False 
        cfg.model.vision_encoder.use_fused_rmsnorm = False
        cfg.model.vision_encoder.use_fused_mlp = False
        
        # 2. Set Base Weights for Initialization
        cfg.model.vision_ckpt_path = base_ckpt_path
        cfg.model.text_ckpt_path = base_ckpt_path
        
        print(f"Initializing MobileViCLIP with base weights: {base_ckpt_path}")
        self.model = MobileViCLIP_Small(cfg, is_pretrain=False)
        
        # 3. Load Fine-Tuned Weights
        print(f"Loading fine-tuned weights: {finetuned_ckpt_path}")
        checkpoint = torch.load(finetuned_ckpt_path, map_location="cpu")
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
            
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Status: {msg}")
        
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def encode_text(self, prompt: str):
        text_input = kinetics_templates[2].format(prompt)
        tokens = self.tokenizer(text_input).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_video_clips(self, clips: torch.Tensor):
        clips = clips.to(self.device)
        with torch.no_grad():
            video_features = self.model.encode_vision(clips, test=True)
            video_features = F.normalize(video_features, dim=-1)
        return video_features

    def find_event(
        self, 
        track_clips: Dict[int, List[Union[torch.Tensor, str]]], 
        prompt: str, 
        clip_stride: int, 
        target_fps: int, 
        clip_duration: float,
        batch_size: int = 8
    ):
        results_timeline = []

        print(f"Searching for event: '{prompt}'")
        text_embeddings = self.encode_text(prompt)

        for track_id, clips_or_paths in track_clips.items():
            if not clips_or_paths: continue

            print(f"Track {track_id}: Analyzing {len(clips_or_paths)} segments...")

            all_scores = []
            
            # --- BATCHED PROCESSING ---
            for i in range(0, len(clips_or_paths), batch_size):
                batch_items = clips_or_paths[i : i + batch_size]
                
                # 1. Load batch (if they are file paths)
                loaded_batch = []
                for item in batch_items:
                    if isinstance(item, str):
                        loaded_batch.append(torch.load(item))
                    else:
                        loaded_batch.append(item)
                
                # 2. Stack and Move to GPU
                batch_tensor = torch.stack(loaded_batch).to(self.device)
                
                # 3. Encode and Score
                video_embs = self.encode_video_clips(batch_tensor)
                
                # [Batch, Dim] @ [Dim, 1] -> [Batch]
                scores = (video_embs @ text_embeddings.T).squeeze(1)
                
                # 4. Save results to CPU and clear GPU memory
                if scores.ndim == 0:
                    all_scores.append([float(scores.cpu().numpy())])
                else:
                    all_scores.append(scores.cpu().numpy())
                    
                del batch_tensor, video_embs, loaded_batch
            
            # Combine all batch results
            scores_np = np.concatenate(all_scores)
            
            # --- Map back to timeline ---
            stride_sec = clip_stride / target_fps

            for idx, score in enumerate(scores_np):
                start_time = idx * stride_sec
                end_time = start_time + clip_duration
                
                if score > 0.25:
                    results_timeline.append({
                        "track_id": track_id,
                        "start": start_time,
                        "end": end_time,
                        "score": float(score)
                    })
        
        return results_timeline
    
    def process_timeline(self, results_timeline, overlap_threshold:float=0.5):
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
            clips.sort(key=lambda x: x['start'])
            
            current_event = None
            
            for clip in clips:
                if current_event is None:
                    current_event = {
                        'track_id': tid,
                        'start': clip['start'],
                        'end': clip['end'],
                        'scores': [clip['score']]
                    }
                else:
                    if clip['start'] <= current_event['end'] + overlap_threshold:
                        current_event['end'] = max(current_event['end'], clip['end'])
                        current_event['scores'].append(clip['score'])
                    else:
                        current_event['score'] = sum(current_event['scores']) / len(current_event['scores'])
                        del current_event['scores']
                        merged_events.append(current_event)
                        
                        current_event = {
                            'track_id': tid,
                            'start': clip['start'],
                            'end': clip['end'],
                            'scores': [clip['score']]
                        }
            
            if current_event:
                current_event['score'] = sum(current_event['scores']) / len(current_event['scores'])
                del current_event['scores']
                merged_events.append(current_event)

        merged_events.sort(key=lambda x: x['score'], reverse=True)
        
        return merged_events