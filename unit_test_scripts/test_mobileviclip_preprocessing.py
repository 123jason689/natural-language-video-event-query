import os
import shutil
import tempfile
import cv2
import numpy as np
import torch
import supervision as sv
from bidict import bidict

# Imports from your library structure
from libs.mobileviclip_preprocessing import MobileViClipPreprocessor
from libs.gdino_process import DetectionResult
from libs.typings_ import ObjectMap

def test_preprocessing_end_to_end():
    """
    Tests the full pipeline from mocked video + detection results 
    to final tensor clips with Temporal Resampling.
    
    Verifies that High-FPS input is correctly resampled to fixed clip_length (8)
    over the specific clip_duration.
    """
    print("="*80)
    print("TEST: MobileViCLIP Preprocessing End-to-End (Temporal Resampling)")
    print("="*80)

    # 1. Setup Temporary Environment
    test_dir = tempfile.mkdtemp()
    video_path = os.path.join(test_dir, "test_video.mp4")
    
    try:
        # 2. Create Dummy Video (High FPS to test resampling)
        # Scenario: 30 FPS video, 4 seconds long -> 120 frames total.
        # MobileViCLIP needs 8 frames covering ~2 seconds. 
        # Without resampling, a 2s clip would be 60 frames (too big for model).
        width, height = 640, 480
        fps = 30
        duration_sec = 4
        num_frames = fps * duration_sec
        
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to create dummy video file.")
            return False

        print(f"Generating dummy video at {video_path}...")
        print(f"  - Duration: {duration_sec}s")
        print(f"  - FPS: {fps}")
        print(f"  - Total Frames: {num_frames}")

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw 'Person' (White square moving left-to-right)
            p_x = (i * 5) % (width - 50)
            cv2.rectangle(frame, (p_x, 100), (p_x + 50, 200), (255, 255, 255), -1)
            
            # Draw 'Car' (Blue rectangle static)
            cv2.rectangle(frame, (300, 300), (450, 400), (255, 0, 0), -1)
            
            out.write(frame)
        out.release()
        
        # 3. Create Mock DetectionResults
        detection_results = []
        
        # Mock ObjectMap
        object_map = ObjectMap()
        object_map.object_map = bidict({'person': 1, 'car': 2})
        
        print("Generating mock detection results for all 120 frames...")
        for i in range(num_frames):
            # Person box (matches video movement)
            p_x = (i * 5) % (width - 50)
            box_person = np.array([p_x, 100, p_x + 50, 200])
            norm_box_person = box_person / np.array([width, height, width, height])
            
            # Car box (static)
            box_car = np.array([300, 300, 450, 400])
            norm_box_car = box_car / np.array([width, height, width, height])
            
            xyxy = np.vstack([norm_box_person, norm_box_car])
            
            # Person has higher confidence (0.9), Car lower (0.8)
            confidence = np.array([0.9, 0.8])
            class_id = np.array([1, 2])     # 1=person, 2=car
            tracker_id = np.array([101, 202]) # Track ID 101, 202
            
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id
            )
            
            result = DetectionResult(
                frame_index=i,
                timestamp_ms=i * (1000/fps), # Accurate timestamp based on FPS
                detections=detections,
                phrases=["person", "car"] 
            )
            detection_results.append(result)

        # 4. Initialize Preprocessor
        # Requirement:
        # - Clip Duration: 2.0 seconds
        # - Output Frames: 8 (Model Requirement)
        # - Stride: 30 frames (1.0 second)
        preprocessor = MobileViClipPreprocessor(
            target_size=224,
            context_padding=0.1,
            clip_length=8,          # T=8 frames output
            clip_duration=2.0,      # T_real=2.0s input window
            clip_stride=18,         # Move window every 1s (30 frames)
            person_keywords=["person"]
        )
        
        # 5. Run Logic
        print("Running preprocessor...")
        output = preprocessor.run(
            video_path,
            detection_results,
            object_map,
            top_k=5 
        )
        
        # 6. Assertions
        
        # A. Check Filtering
        if 101 not in output:
            print(f"❌ Track ID 101 (Person) missing. Keys: {list(output.keys())}")
            return False
        if 202 in output:
            print("❌ Track ID 202 (Car) should have been filtered out.")
            return False
        
        print("✅ Class filtering successful.")
        
        # B. Check Temporal Windowing
        # Video is 4s. Clip duration 2s. Stride 1s.
        # Windows: [0-2s], [1-3s], [2-4s]. 
        # Expected: 3 clips (maybe 4 if end-padding logic catches the tail).
        person_clips = output[101]
        
        print(f"✅ Generated {len(person_clips)} clips.")
        if len(person_clips) < 2:
             print("❌ Unexpectedly low number of clips.")
             return False

        # C. Check Tensor Structure (The most important check for the fix)
        sample_tensor = person_clips[0]
        
        # Expected shape: (C, T, H, W) -> (3, 8, 224, 224)
        # If the fix FAILED, T would be 60 (2 seconds * 30 FPS)
        # If the fix PASSED, T would be 8 (resampled)
        expected_shape = (3, 8, 224, 224)
        
        if sample_tensor.shape != expected_shape:
            print(f"❌ Tensor shape mismatch.")
            print(f"   Expected: {expected_shape}")
            print(f"   Got:      {sample_tensor.shape}")
            print("   (This likely means temporal resampling is NOT working)")
            return False
            
        if not isinstance(sample_tensor, torch.Tensor):
             print("❌ Output is not a torch.Tensor")
             return False
             
        # Check normalization
        if sample_tensor.max() > 1.0 or sample_tensor.min() < 0.0:
             print(f"❌ Values not normalized: range [{sample_tensor.min():.2f}, {sample_tensor.max():.2f}]")
             return False

        print(f"✅ Tensor shape verified: {sample_tensor.shape}")
        print("   (60 input frames successfully resampled to 8 output frames)")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    print("\n✅ TEST PASSED: MobileViCLIP Preprocessing")
    print("="*80)
    return True

if __name__ == "__main__":
    test_preprocessing_end_to_end()