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
    to final tensor clips.
    """
    print("="*80)
    print("TEST: MobileViCLIP Preprocessing End-to-End")
    print("="*80)

    # 1. Setup Temporary Environment
    test_dir = tempfile.mkdtemp()
    video_path = os.path.join(test_dir, "test_video.mp4")
    
    try:
        # 2. Create Dummy Video (20 frames, 640x480)
        # We simulate a "Person" moving left-to-right and a "Car" staying static.
        width, height = 640, 480
        fps = 10
        num_frames = 20
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to create dummy video file.")
            return False

        print(f"Generating dummy video at {video_path}...")
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw 'Person' (White square moving)
            p_x = 100 + i * 10
            cv2.rectangle(frame, (p_x, 100), (p_x + 50, 200), (255, 255, 255), -1)
            
            # Draw 'Car' (Blue rectangle static)
            cv2.rectangle(frame, (300, 300), (450, 400), (255, 0, 0), -1)
            
            out.write(frame)
        out.release()
        
        # 3. Create Mock DetectionResults
        # Track ID 1: Person (detected in frames 0-19)
        # Track ID 2: Car (detected in frames 0-19)
        detection_results = []
        
        # Mock ObjectMap
        object_map = ObjectMap()
        # bidict required: string <-> int
        object_map.object_map = bidict({'person': 1, 'car': 2})
        
        print("Generating mock detection results...")
        for i in range(num_frames):
            # Normalized boxes [x1, y1, x2, y2]
            
            # Person box (matches video movement)
            p_x = 100 + i * 10
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
                timestamp_ms=i * (1000/fps),
                detections=detections,
                phrases=["person", "car"] 
            )
            detection_results.append(result)

        # 4. Initialize Preprocessor
        # Config: Clip length 4, Stride 4. 20 frames -> 5 clips.
        preprocessor = MobileViClipPreprocessor(
            target_size=224,
            context_padding=0.1,
            clip_length=4,
            clip_stride=4,
            person_keywords=["person"] # Should match 'person' in object_map
        )
        
        # 5. Run Logic
        print("Running preprocessor...")
        output = preprocessor.run(
            video_path,
            detection_results,
            object_map,
            top_k=5 # Get top 5 to ensure both would be selected if not for filtering
        )
        
        # 6. Assertions
        
        # A. Check Filtering
        if 101 not in output:
            print(f"❌ Track ID 101 (Person) missing from output. Keys found: {list(output.keys())}")
            return False
            
        if 202 in output:
            print("❌ Track ID 202 (Car) should have been filtered out (not a person).")
            return False
        
        print("✅ Class filtering successful (Person kept, Car ignored).")
        
        # B. Check Tensor Structure
        person_clips = output[101]
        expected_clips = num_frames // 4 # 20 // 4 = 5
        
        if len(person_clips) != expected_clips:
            print(f"❌ Unexpected number of clips. Expected {expected_clips}, got {len(person_clips)}")
            return False
            
        print(f"✅ Generated {len(person_clips)} clips (matches expected).")
        
        sample_tensor = person_clips[0]
        # Expected shape: (C, T, H, W) -> (3, 4, 224, 224)
        expected_shape = (3, 4, 224, 224)
        
        if sample_tensor.shape != expected_shape:
            print(f"❌ Tensor shape mismatch. Expected {expected_shape}, got {sample_tensor.shape}")
            return False
            
        if not isinstance(sample_tensor, torch.Tensor):
             print("❌ Output is not a torch.Tensor")
             return False
             
        # Check value range (should be normalized 0-1)
        if sample_tensor.max() > 1.0 or sample_tensor.min() < 0.0:
             print("❌ Tensor values not normalized to [0, 1]")
             return False

        print(f"✅ Tensor shape verification successful: {sample_tensor.shape}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    print("\n✅ TEST PASSED: MobileViCLIP Preprocessing")
    print("="*80)
    return True

if __name__ == "__main__":
    test_preprocessing_end_to_end()