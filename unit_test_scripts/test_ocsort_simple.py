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
    """Test tracking across multiple frames."""
    print("\n" + "="*60)
    print("TEST: Multi-Frame Tracking")
    print("="*60)
    
    tracker = OCSort(
        det_thresh=0.3,
        max_age=30,
        min_hits=1,  # Set to 1 to see results immediately
        iou_threshold=0.3
    )
    
    img_h, img_w = 480, 640
    
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
              f"score={detections[0,4]:.2f}")
        
        output = tracker.update(detections, (img_h, img_w), (img_h, img_w))
        
        if len(output) > 0:
            print(f"Output: tracker_id={int(output[0,6])}, "
                  f"box=[{output[0,0]:.0f}, {output[0,1]:.0f}, "
                  f"{output[0,2]:.0f}, {output[0,3]:.0f}]")
        else:
            print(f"Output: (no tracked objects yet)")
    
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
    test_input_validation()
    
    print("\n✅ All tests completed\n")
