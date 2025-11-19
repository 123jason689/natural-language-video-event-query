"""
Simple unit test for OC-Sort update method.

This is a minimal test to verify the basic input/output format
without dependencies on the full GDINO pipeline.
"""

import numpy as np
from libs.ocsort.ocsort import OCSort

def test_basic_update():
    """Test basic OC-Sort update with proper format."""
    print("="*60)
    print("TEST: Basic OC-Sort Update")
    print("="*60)
    
    # Initialize tracker
    tracker = OCSort(
        det_thresh=0.3,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Image size
    img_h, img_w = 480, 640
    
    # Create sample detections: [x1, y1, x2, y2, score, class_id]
    detections = np.array([
        [100, 100, 200, 200, 0.9, 0],  # Person
        [300, 150, 400, 250, 0.85, 1], # Car
        [450, 200, 550, 350, 0.75, 0], # Person
    ])
    
    print(f"\nInput detections:")
    print(f"  Shape: {detections.shape}")
    print(f"  Format: [x1, y1, x2, y2, score, class_id]")
    print(f"  Data:\n{detections}")
    
    # Update tracker
    output = tracker.update(
        detections,
        img_info=(img_h, img_w),
        img_size=(img_h, img_w)
    )
    
    print(f"\nOutput from OC-Sort:")
    print(f"  Shape: {output.shape}")
    print(f"  Format: [x1, y1, x2, y2, score, class_id, tracker_id]")
    if len(output) > 0:
        print(f"  Data:\n{output}")
    else:
        print(f"  (Empty - objects not confirmed yet, need min_hits=3)")
    
    print("\n" + "="*60)
    return output


def test_multi_frame_tracking():
    """Test tracking a single object across multiple frames."""
    print("\n" + "="*60)
    print("TEST: Single Object Multi-Frame Tracking")
    print("="*60)
    
    tracker = OCSort(
        det_thresh=0.3,
        max_age=30,
        min_hits=1,  # Set to 1 to see results immediately
        iou_threshold=0.3
    )
    
    img_h, img_w = 480, 640
    
    print("\nSimulating a single object moving to the right across 5 frames...")
    
    # Track the tracker ID to verify consistency
    tracker_ids = []
    
    # Simulate 5 frames with moving object
    for frame_num in range(1, 6):
        print(f"\n--- Frame {frame_num} ---")
        
        # Object moving to the right
        x_offset = (frame_num - 1) * 20
        detections = np.array([
            [100 + x_offset, 100, 200 + x_offset, 200, 0.9, 0],
        ])
        
        print(f"Input: box=[{detections[0,0]:.0f}, {detections[0,1]:.0f}, "
              f"{detections[0,2]:.0f}, {detections[0,3]:.0f}], "
              f"score={detections[0,4]:.2f}, class_id={int(detections[0,5])}")
        
        output = tracker.update(detections, (img_h, img_w), (img_h, img_w))
        
        if len(output) > 0:
            tracker_id = int(output[0,6])
            tracker_ids.append(tracker_id)
            print(f"Output: tracker_id={tracker_id}, "
                  f"box=[{output[0,0]:.0f}, {output[0,1]:.0f}, "
                  f"{output[0,2]:.0f}, {output[0,3]:.0f}], "
                  f"score={output[0,4]:.2f}, class_id={int(output[0,5])}")
        else:
            print(f"Output: (no tracked objects yet)")
    
    # Verify tracking consistency
    if len(tracker_ids) > 0:
        unique_ids = set(tracker_ids)
        if len(unique_ids) == 1:
            print(f"\n✅ Tracking consistent: Same tracker ID {tracker_ids[0]} across all frames")
        else:
            print(f"\n⚠️  Warning: Multiple tracker IDs detected: {unique_ids}")
            raise AssertionError(f"Expected consistent tracker ID, got {unique_ids}")
    
    print("\n" + "="*60)


def test_multi_object_tracking():
    """Test tracking multiple objects across multiple frames."""
    print("\n" + "="*60)
    print("TEST: Multi-Object Multi-Frame Tracking")
    print("="*60)
    
    tracker = OCSort(
        det_thresh=0.3,
        max_age=30,
        min_hits=1,  # Set to 1 to see results immediately
        iou_threshold=0.3
    )
    
    img_h, img_w = 480, 640
    
    print("\nSimulating 3 objects with different behaviors across 10 frames:")
    print("  • Object 1 (person): Moving right steadily")
    print("  • Object 2 (car): Moving down")
    print("  • Object 3 (bicycle): Stationary, disappears at frame 6")
    
    # Dictionary to track IDs for each object type
    object_tracks = {
        'person': [],
        'car': [],
        'bicycle': []
    }
    
    for frame_num in range(1, 11):
        print(f"\n{'─'*60}")
        print(f"Frame {frame_num}/10")
        print(f"{'─'*60}")
        
        detections = []
        
        # Object 1: Person moving right
        x_offset = (frame_num - 1) * 15
        detections.append([100 + x_offset, 100, 200 + x_offset, 200, 0.92, 0])
        
        # Object 2: Car moving down
        y_offset = (frame_num - 1) * 10
        detections.append([300, 50 + y_offset, 400, 150 + y_offset, 0.88, 1])
        
        # Object 3: Bicycle (stationary, disappears after frame 5)
        if frame_num <= 5:
            detections.append([450, 200, 550, 300, 0.75, 2])
        
        detections = np.array(detections)
        
        print(f"Input: {len(detections)} detections")
        for i, det in enumerate(detections):
            class_names = ['person', 'car', 'bicycle']
            class_name = class_names[int(det[5])] if int(det[5]) < len(class_names) else f"class_{int(det[5])}"
            print(f"  Det {i+1}: {class_name:8s} box=[{det[0]:3.0f},{det[1]:3.0f},{det[2]:3.0f},{det[3]:3.0f}] score={det[4]:.2f}")
        
        output = tracker.update(detections, (img_h, img_w), (img_h, img_w))
        
        print(f"Output: {len(output)} tracked objects")
        for i, track in enumerate(output):
            class_names = ['person', 'car', 'bicycle']
            class_id = int(track[5])
            tracker_id = int(track[6])
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Record tracker ID for consistency check
            object_tracks[class_name].append(tracker_id)
            
            print(f"  Track {i+1}: {class_name:8s} id={tracker_id} "
                  f"box=[{track[0]:3.0f},{track[1]:3.0f},{track[2]:3.0f},{track[3]:3.0f}] "
                  f"score={track[4]:.2f}")
    
    # Verify tracking consistency
    print(f"\n{'═'*60}")
    print("Tracking Consistency Analysis:")
    print(f"{'═'*60}")
    
    all_consistent = True
    for obj_type, tracker_ids in object_tracks.items():
        if len(tracker_ids) > 0:
            unique_ids = set(tracker_ids)
            if len(unique_ids) == 1:
                print(f"✅ {obj_type:8s}: Consistent tracker ID {list(unique_ids)[0]} across {len(tracker_ids)} frames")
            else:
                print(f"⚠️  {obj_type:8s}: Multiple IDs {unique_ids} across {len(tracker_ids)} frames")
                all_consistent = False
        else:
            print(f"ℹ️  {obj_type:8s}: No tracks recorded")
    
    if not all_consistent:
        raise AssertionError("Tracking IDs were not consistent across frames")
    
    # Verify bicycle disappeared correctly
    bicycle_count = len(object_tracks['bicycle'])
    if bicycle_count > 5:
        print(f"\n⚠️  Warning: Bicycle tracked for {bicycle_count} frames, expected ≤5")
    else:
        print(f"\n✅ Bicycle correctly stopped tracking after disappearing (tracked for {bicycle_count} frames)")
    
    print("\n" + "="*60)


def test_input_validation():
    """Test various input shapes and formats."""
    print("\n" + "="*60)
    print("TEST: Input Validation")
    print("="*60)
    
    tracker = OCSort(det_thresh=0.3)
    img_h, img_w = 480, 640
    
    test_cases = [
        ("Empty detections", np.empty((0, 6))),
        ("Single detection", np.array([[100, 100, 200, 200, 0.9, 0]])),
        ("Multiple detections", np.array([
            [100, 100, 200, 200, 0.9, 0],
            [300, 150, 400, 250, 0.85, 1],
        ])),
    ]
    
    for name, dets in test_cases:
        print(f"\n{name}:")
        print(f"  Input shape: {dets.shape}")
        try:
            output = tracker.update(dets, (img_h, img_w), (img_h, img_w))
            print(f"  Output shape: {output.shape}")
            print(f"  ✅ Success")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("\n🧪 OC-SORT SIMPLE TEST SUITE\n")
    
    test_basic_update()
    test_multi_frame_tracking()
    test_multi_object_tracking()
    test_input_validation()
    
    print("\n✅ All tests completed\n")
