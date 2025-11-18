"""
Test to demonstrate the torch.range() bug in gdino_process.py

This test shows the shape mismatch caused by using torch.range() 
instead of torch.arange() in the post_process_result method.
"""

import torch
import numpy as np


def demonstrate_torch_range_bug():
    """
    Demonstrates the bug in gdino_process.py line 175
    """
    print("="*70)
    print("DEMONSTRATION: torch.range() vs torch.arange() Bug")
    print("="*70)
    
    # Simulate having 5 detections
    num_detections = 5
    
    print(f"\nScenario: We have {num_detections} detections")
    print(f"We want to create class indices: [0, 1, 2, 3, 4]")
    
    # Current buggy implementation in gdino_process.py
    print("\n1. Current implementation (BUGGY):")
    print(f"   phrase_class_idx = torch.range(0, xyxy.shape[0]).numpy()")
    
    buggy_indices = torch.range(0, num_detections).numpy()
    print(f"   Result: {buggy_indices}")
    print(f"   Shape: {buggy_indices.shape}")
    print(f"   ❌ Problem: Creates {len(buggy_indices)} elements instead of {num_detections}!")
    
    # Correct implementation
    print("\n2. Correct implementation:")
    print(f"   phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()")
    
    correct_indices = torch.arange(num_detections).numpy()
    print(f"   Result: {correct_indices}")
    print(f"   Shape: {correct_indices.shape}")
    print(f"   ✅ Correct: Creates exactly {num_detections} elements")
    
    # Show the impact on np.column_stack
    print("\n3. Impact on np.column_stack([xyxy, confidence, phrase_class_idx]):")
    
    # Mock data
    xyxy = np.random.rand(num_detections, 4) * 100
    confidence = np.random.rand(num_detections)
    
    print(f"   xyxy shape: {xyxy.shape}")
    print(f"   confidence shape: {confidence.shape}")
    
    try:
        print(f"\n   Using buggy indices (shape {buggy_indices.shape}):")
        buggy_result = np.column_stack([xyxy, confidence, buggy_indices])
        print(f"   ❌ Result shape: {buggy_result.shape}")
        print(f"      Last row has extra element: {buggy_result[-1]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        print(f"\n   Using correct indices (shape {correct_indices.shape}):")
        correct_result = np.column_stack([xyxy, confidence, correct_indices])
        print(f"   ✅ Result shape: {correct_result.shape}")
        print(f"      Each row has 6 elements (x1,y1,x2,y2,score,class_id)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)


def test_ocsort_with_wrong_shape():
    """
    Test what happens when we pass the wrong shape to OC-Sort
    """
    print("\n" + "="*70)
    print("TEST: What happens with wrong input shape?")
    print("="*70)
    
    from libs.ocsort.ocsort import OCSort
    
    tracker = OCSort(det_thresh=0.3)
    img_h, img_w = 480, 640
    
    # Correct shape (6 columns)
    print("\n1. Correct input shape (N, 6):")
    correct_input = np.array([
        [100, 100, 200, 200, 0.9, 0],
        [300, 150, 400, 250, 0.85, 1],
    ])
    print(f"   Shape: {correct_input.shape}")
    try:
        result = tracker.update(correct_input, (img_h, img_w), (img_h, img_w))
        print(f"   ✅ Works fine, output shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Wrong shape (7 columns) - what torch.range() would cause
    print("\n2. Wrong input shape (N, 7) - extra column:")
    # Reset tracker
    tracker = OCSort(det_thresh=0.3)
    wrong_input = np.array([
        [100, 100, 200, 200, 0.9, 0, 999],  # Extra column
        [300, 150, 400, 250, 0.85, 1, 999],
    ])
    print(f"   Shape: {wrong_input.shape}")
    try:
        result = tracker.update(wrong_input, (img_h, img_w), (img_h, img_w))
        print(f"   Result: {result.shape}")
        print(f"   ⚠️  Warning: OC-Sort might handle this incorrectly")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)


def show_fix_recommendation():
    """
    Show the recommended fix
    """
    print("\n" + "="*70)
    print("RECOMMENDED FIX")
    print("="*70)
    
    print("\nIn gdino_process.py, line 175:")
    print("\nCurrent (BUGGY):")
    print("  phrase_class_idx = torch.range(0, xyxy.shape[0]).numpy()")
    
    print("\nShould be changed to:")
    print("  phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()")
    
    print("\nOr alternatively:")
    print("  phrase_class_idx = np.arange(xyxy.shape[0])")
    
    print("\nWhy?")
    print("  • torch.range(0, N) is INCLUSIVE and creates [0, 1, 2, ..., N] (N+1 elements)")
    print("  • torch.arange(N) creates [0, 1, 2, ..., N-1] (N elements)")
    print("  • torch.range() is also deprecated since PyTorch 0.5.0")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "🐛 "*20)
    print("BUG DEMONSTRATION: torch.range() in gdino_process.py")
    print("🐛 "*20 + "\n")
    
    demonstrate_torch_range_bug()
    test_ocsort_with_wrong_shape()
    show_fix_recommendation()
    
    print("\n")
