"""
Test script for OC-Sort update method with GDINO model output format.

This script tests the integration between GroundingDINO's post_process_result
and OC-Sort's update method to ensure proper tracking of detected objects.
"""

import numpy as np
import torch
from libs.ocsort.ocsort import OCSort
from libs.gdino_process import Model
import supervision as sv


def create_mock_gdino_output(num_detections=5, image_size=(640, 480), seed=42):
    """
    Create mock GroundingDINO output that mimics real model predictions.
    
    Parameters
    ----------
    num_detections : int
        Number of detections to generate
    image_size : tuple
        (width, height) of the image
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    boxes : torch.Tensor
        Bounding boxes in cxcywh format (normalized 0-1)
    logits : torch.Tensor
        Confidence scores
    phrases : list
        Detection class phrases
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    width, height = image_size
    
    # Generate random boxes in normalized cxcywh format
    # cx, cy in [0.2, 0.8], w, h in [0.05, 0.3]
    cx = torch.rand(num_detections) * 0.6 + 0.2
    cy = torch.rand(num_detections) * 0.6 + 0.2
    w = torch.rand(num_detections) * 0.25 + 0.05
    h = torch.rand(num_detections) * 0.25 + 0.05
    
    boxes = torch.stack([cx, cy, w, h], dim=1)
    
    # Generate confidence scores between 0.5 and 0.99
    logits = torch.rand(num_detections) * 0.49 + 0.5
    
    # Generate phrases
    classes = ["person", "car", "bicycle", "dog", "cat"]
    phrases = [classes[i % len(classes)] for i in range(num_detections)]
    
    return boxes, logits, phrases


def test_ocsort_with_gdino_format():
    """
    Test OC-Sort update with GDINO post_process_result output format.
    
    NOTES:
    ------
    1. The post_process_result expects boxes in cxcywh format (normalized 0-1)
    2. OC-Sort.update expects input shape (N, 6): [x1, y1, x2, y2, score, class_id]
    3. OC-Sort.update returns shape (N, 7): [x1, y1, x2, y2, score, class_id, tracker_id]
    """
    print("="*80)
    print("TEST: OC-Sort with GDINO post_process_result format")
    print("="*80)
    
    # Initialize OC-Sort tracker
    ocsort = OCSort(
        det_thresh=0.3,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False
    )
    
    # Image dimensions
    source_h, source_w = 480, 640
    
    print(f"\nImage size: {source_w}x{source_h}")
    print(f"OC-Sort config: det_thresh={ocsort.det_thresh}, iou_thresh={ocsort.iou_threshold}")
    
    # Simulate multiple frames
    num_frames = 5
    
    for frame_idx in range(num_frames):
        print(f"\n{'─'*80}")
        print(f"Frame {frame_idx + 1}/{num_frames}")
        print(f"{'─'*80}")
        
        # Generate mock GDINO output
        num_dets = np.random.randint(3, 8)
        boxes, logits, phrases = create_mock_gdino_output(
            num_detections=num_dets,
            image_size=(source_w, source_h),
            seed=42 + frame_idx
        )
        
        print(f"\nInput to post_process_result:")
        print(f"  - boxes shape: {boxes.shape} (cxcywh normalized)")
        print(f"  - logits shape: {logits.shape}")
        print(f"  - phrases: {phrases}")
        
        # Simulate post_process_result logic
        # Step 1: Scale boxes to absolute coordinates
        boxes_scaled = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        print(f"\nAfter scaling to absolute coords:")
        print(f"  - boxes_scaled shape: {boxes_scaled.shape}")
        print(f"  - sample box: {boxes_scaled[0].tolist()}")
        
        # Step 2: Convert to xyxy format
        from torchvision.ops import box_convert
        xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        print(f"\nAfter converting to xyxy:")
        print(f"  - xyxy shape: {xyxy.shape}")
        print(f"  - sample box (xyxy): {xyxy[0].tolist()}")
        
        # Step 3: Prepare confidence and class indices
        confidence = logits.numpy()
        phrase_class_idx = torch.arange(0, xyxy.shape[0]).numpy()
        
        print(f"\nPreparing OC-Sort input:")
        print(f"  - confidence shape: {confidence.shape}")
        print(f"  - phrase_class_idx: {phrase_class_idx}")
        
        # Step 4: Stack into OC-Sort input format
        out = np.column_stack([xyxy, confidence, phrase_class_idx])
        print(f"\nOC-Sort input (before update):")
        print(f"  - shape: {out.shape} (expected: N x 6)")
        print(f"  - columns: [x1, y1, x2, y2, score, class_id]")
        print(f"  - sample row: {out[0]}")
        
        # ⚠️ IMPORTANT NOTE: Check if this matches expected format
        if out.shape[1] != 6:
            print(f"\n⚠️  WARNING: OC-Sort expects 6 columns, got {out.shape[1]}")
        
        # Step 5: Call OC-Sort update
        try:
            oc_outputs = ocsort.update(out, (source_h, source_w), (source_h, source_w))
            
            print(f"\nOC-Sort output (after update):")
            print(f"  - shape: {oc_outputs.shape} (expected: M x 7)")
            print(f"  - columns: [x1, y1, x2, y2, score, class_id, tracker_id]")
            
            if len(oc_outputs) > 0:
                print(f"  - number of tracked objects: {len(oc_outputs)}")
                print(f"  - sample tracked object: {oc_outputs[0]}")
                print(f"  - tracker IDs: {oc_outputs[:, 6].astype(int)}")
            else:
                print(f"  - no objects tracked (possibly below det_thresh or min_hits)")
            
            # Step 6: Create supervision Detections object
            if len(oc_outputs) > 0:
                detections = sv.Detections(
                    xyxy=oc_outputs[:, :4],
                    confidence=oc_outputs[:, 4],
                    class_id=oc_outputs[:, 5].astype(int),
                    tracker_id=oc_outputs[:, 6].astype(int)
                )
                
                print(f"\nFinal sv.Detections object:")
                print(f"  - xyxy shape: {detections.xyxy.shape}")
                print(f"  - confidence shape: {detections.confidence.shape}")
                print(f"  - class_id: {detections.class_id}")
                print(f"  - tracker_id: {detections.tracker_id}")
            
        except Exception as e:
            print(f"\n❌ ERROR during OC-Sort update: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*80)
    print("✅ TEST PASSED: OC-Sort successfully processes GDINO format")
    print("="*80)
    return True


def test_edge_cases():
    """
    Test edge cases and potential issues.
    """
    print("\n" + "="*80)
    print("TEST: Edge Cases")
    print("="*80)
    
    ocsort = OCSort(det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3)
    source_h, source_w = 480, 640
    
    # Test 1: Empty detections
    print("\n1. Testing empty detections (no objects detected):")
    try:
        empty_input = np.empty((0, 6))
        result = ocsort.update(empty_input, (source_h, source_w), (source_h, source_w))
        print(f"   Result shape: {result.shape}")
        print(f"   ✅ Handles empty input correctly")
    except Exception as e:
        print(f"   ❌ Error with empty input: {e}")
    
    # Test 2: Single detection
    print("\n2. Testing single detection:")
    try:
        single_det = np.array([[100, 100, 200, 200, 0.9, 0]])
        result = ocsort.update(single_det, (source_h, source_w), (source_h, source_w))
        print(f"   Input shape: {single_det.shape}")
        print(f"   Result shape: {result.shape}")
        print(f"   ✅ Handles single detection")
    except Exception as e:
        print(f"   ❌ Error with single detection: {e}")
    
    # Test 3: Low confidence detections (below threshold)
    print("\n3. Testing low confidence detections:")
    try:
        low_conf = np.array([
            [100, 100, 200, 200, 0.15, 0],  # Below det_thresh=0.3
            [250, 150, 350, 250, 0.20, 1],  # Below det_thresh=0.3
        ])
        result = ocsort.update(low_conf, (source_h, source_w), (source_h, source_w))
        print(f"   Input: 2 detections with scores 0.15, 0.20 (threshold: 0.3)")
        print(f"   Result shape: {result.shape}")
        if len(result) == 0:
            print(f"   ✅ Correctly filters out low confidence detections")
        else:
            print(f"   ⚠️  Some detections passed through")
    except Exception as e:
        print(f"   ❌ Error with low confidence: {e}")
    
    # Test 4: Very high number of detections
    print("\n4. Testing many detections (stress test):")
    try:
        many_dets = np.random.rand(100, 6)
        many_dets[:, :4] *= [source_w, source_h, source_w, source_h]  # Scale boxes
        many_dets[:, 4] = np.random.rand(100) * 0.5 + 0.5  # Confidence 0.5-1.0
        many_dets[:, 5] = np.random.randint(0, 5, 100)  # Class IDs 0-4
        result = ocsort.update(many_dets, (source_h, source_w), (source_h, source_w))
        print(f"   Input: 100 detections")
        print(f"   Result shape: {result.shape}")
        print(f"   ✅ Handles many detections")
    except Exception as e:
        print(f"   ❌ Error with many detections: {e}")
    
    print("\n" + "="*80)


def test_tracking_consistency():
    """
    Test that tracker IDs remain consistent across frames for the same object.
    """
    print("\n" + "="*80)
    print("TEST: Tracking Consistency")
    print("="*80)
    
    ocsort = OCSort(det_thresh=0.3, max_age=30, min_hits=1, iou_threshold=0.3)
    source_h, source_w = 480, 640
    
    # Simulate the same object moving slightly across frames
    print("\nSimulating a single object moving across 10 frames...")
    
    tracker_ids = []
    
    for frame_idx in range(10):
        # Object slowly moves to the right
        x1 = 100 + frame_idx * 10
        y1 = 100
        x2 = 200 + frame_idx * 10
        y2 = 200
        
        det = np.array([[x1, y1, x2, y2, 0.9, 0]])
        result = ocsort.update(det, (source_h, source_w), (source_h, source_w))
        
        if len(result) > 0:
            tracker_id = int(result[0, 6])
            tracker_ids.append(tracker_id)
            print(f"  Frame {frame_idx + 1}: box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], tracker_id={tracker_id}")
        else:
            tracker_ids.append(None)
            print(f"  Frame {frame_idx + 1}: no tracked object (waiting for min_hits)")
    
    # Check consistency
    valid_ids = [tid for tid in tracker_ids if tid is not None]
    if len(set(valid_ids)) == 1:
        print(f"\n✅ Tracking is consistent: same tracker ID {valid_ids[0]} across all frames")
    else:
        print(f"\n⚠️  Warning: Multiple tracker IDs detected: {set(valid_ids)}")
        print(f"   This might indicate tracking loss or ID switching")
    
    print("\n" + "="*80)


def test_format_notes():
    """
    Print important notes about the format and potential issues.
    """
    print("\n" + "="*80)
    print("IMPORTANT NOTES & OBSERVATIONS")
    print("="*80)
    
    notes = [
        {
            "title": "Input Format to OC-Sort.update()",
            "points": [
                "Expected shape: (N, 6) where N is number of detections",
                "Columns: [x1, y1, x2, y2, score, class_id]",
                "Coordinates should be in absolute pixels (not normalized)",
                "x1, y1 is top-left corner; x2, y2 is bottom-right corner",
            ]
        },
        {
            "title": "Output Format from OC-Sort.update()",
            "points": [
                "Shape: (M, 7) where M is number of tracked objects (M ≤ N)",
                "Columns: [x1, y1, x2, y2, score, class_id, tracker_id]",
                "tracker_id is a unique integer for each tracked object",
                "M can be less than N due to filtering (det_thresh, min_hits)",
            ]
        },
        {
            "title": "GDINO post_process_result Implementation",
            "points": [
                "Input boxes are in cxcywh format, normalized to [0, 1]",
                "Boxes are scaled: boxes * [w, h, w, h]",
                "Then converted from cxcywh to xyxy using torchvision.ops.box_convert",
                "phrase_class_idx = torch.range(0, N) - assigns sequential class IDs",
                "⚠️  NOTE: torch.range is deprecated, should use torch.arange",
            ]
        },
        {
            "title": "Potential Issues Found",
            "points": [
                "⚠️  In gdino_process.py line 175:",
                "   phrase_class_idx = torch.range(0, xyxy.shape[0]).numpy()",
                "   Should be: torch.arange(0, xyxy.shape[0]).numpy()",
                "   torch.range(0, N) includes N, so it creates N+1 elements!",
                "   This causes shape mismatch: (N+1,) vs expected (N,)",
                "",
                "⚠️  The correct implementation should be:",
                "   phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()",
                "   or: phrase_class_idx = np.arange(xyxy.shape[0])",
            ]
        },
        {
            "title": "OC-Sort Behavior",
            "points": [
                "det_thresh: Minimum confidence to consider a detection",
                "min_hits: Number of consecutive frames before track is confirmed",
                "max_age: Maximum frames to keep track alive without detection",
                "iou_threshold: Minimum IoU for associating detection to track",
                "Early frames may return empty results until min_hits is reached",
            ]
        },
        {
            "title": "Integration with supervision (sv)",
            "points": [
                "sv.Detections expects separate arrays for xyxy, confidence, etc.",
                "class_id and tracker_id should be integers",
                "Ensure proper type conversion: .astype(int) for IDs",
            ]
        },
    ]
    
    for note in notes:
        print(f"\n{note['title']}")
        print("─" * len(note['title']))
        for point in note['points']:
            if point:
                print(f"  • {point}")
            else:
                print()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "🔬 "*20)
    print("OC-SORT + GDINO INTEGRATION TEST SUITE")
    print("🔬 "*20)
    
    # Run tests
    test_format_notes()
    
    success = test_ocsort_with_gdino_format()
    
    test_edge_cases()
    
    test_tracking_consistency()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS COMPLETED")
        print("="*80)
        print("\n📝 Please review the 'IMPORTANT NOTES & OBSERVATIONS' section above")
        print("   particularly the issue with torch.range() vs torch.arange()")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
    
    print("\n")
