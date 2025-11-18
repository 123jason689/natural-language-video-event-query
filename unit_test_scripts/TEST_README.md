# OC-Sort + GDINO Integration Tests

This directory contains test scripts for validating the integration between GroundingDINO and OC-Sort tracking.

## Test Files

### 1. `test_ocsort_simple.py`
**Purpose**: Basic unit tests for OC-Sort functionality

**What it tests**:
- Basic OC-Sort update with proper input format
- Multi-frame tracking with moving objects
- Input validation with various shapes

**Run it**:
```bash
python test_ocsort_simple.py
```

**Good for**: Quick validation that OC-Sort is working correctly

---

### 2. `test_ocsort_gdino_integration.py`
**Purpose**: Comprehensive integration test simulating the full GDINO → OC-Sort pipeline

**What it tests**:
- Complete flow from GDINO boxes (cxcywh normalized) to OC-Sort output
- Format conversions at each step
- Edge cases (empty detections, single object, many objects, low confidence)
- Tracking consistency across frames
- supervision.Detections object creation

**Run it**:
```bash
python test_ocsort_gdino_integration.py
```

**Good for**: Understanding the complete data flow and identifying format issues

---

### 3. `test_torch_range_bug.py`
**Purpose**: Demonstrates a critical bug in `gdino_process.py`

**What it shows**:
- The difference between `torch.range()` and `torch.arange()`
- How `torch.range(0, N)` creates N+1 elements instead of N
- Impact on `np.column_stack()` operation
- Why OC-Sort might receive wrong input shape

**Run it**:
```bash
python test_torch_range_bug.py
```

**Good for**: Understanding why there might be shape mismatches

---

## Critical Issue Found ⚠️

**Location**: `libs/gdino_process.py`, line 175

**Current code**:
```python
phrase_class_idx = torch.range(0, xyxy.shape[0]).numpy()
```

**Problem**: 
- `torch.range(0, N)` is **inclusive** and creates `[0, 1, 2, ..., N]` → **(N+1) elements**
- This creates a shape mismatch when stacking with `xyxy` which has N rows
- `torch.range()` is also **deprecated** since PyTorch 0.5.0

**Recommended fix**:
```python
phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()
```
or
```python
phrase_class_idx = np.arange(xyxy.shape[0])
```

**Why this matters**:
- OC-Sort expects input shape `(N, 6)`: `[x1, y1, x2, y2, score, class_id]`
- Wrong shape will cause errors or incorrect tracking

---

## Expected Data Flow

### 1. GDINO Output → post_process_result Input
```python
boxes:  torch.Tensor, shape (N, 4), format: cxcywh, normalized [0, 1]
logits: torch.Tensor, shape (N,)
phrases: List[str], length N
```

### 2. Inside post_process_result
```python
# Step 1: Scale to absolute coordinates
boxes_scaled = boxes * torch.Tensor([w, h, w, h])

# Step 2: Convert format
xyxy = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()

# Step 3: Prepare OC-Sort input
confidence = logits.numpy()
phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()  # FIX THIS!
out = np.column_stack([xyxy, confidence, phrase_class_idx])
# Result: (N, 6) array
```

### 3. OC-Sort Input
```python
Shape: (N, 6)
Format: [x1, y1, x2, y2, score, class_id]
- x1, y1: top-left corner (absolute pixels)
- x2, y2: bottom-right corner (absolute pixels)
- score: confidence score [0, 1]
- class_id: integer class identifier
```

### 4. OC-Sort Output
```python
Shape: (M, 7)  where M ≤ N
Format: [x1, y1, x2, y2, score, class_id, tracker_id]
- First 6 columns same as input
- tracker_id: unique integer for each tracked object
```

### 5. supervision.Detections
```python
sv.Detections(
    xyxy=output[:, :4],           # (M, 4)
    confidence=output[:, 4],      # (M,)
    class_id=output[:, 5].astype(int),    # (M,)
    tracker_id=output[:, 6].astype(int)   # (M,)
)
```

---

## Running All Tests

```bash
# Simple tests
python test_ocsort_simple.py

# Comprehensive integration tests
python test_ocsort_gdino_integration.py

# Bug demonstration
python test_torch_range_bug.py
```

---

## OC-Sort Configuration

Important parameters in `OCSort.__init__()`:

- **`det_thresh`** (default: 0.3): Minimum confidence score to consider a detection
- **`min_hits`** (default: 3): Number of consecutive frames before a track is confirmed
  - Objects won't appear in output until they've been detected `min_hits` times
  - Set to 1 for immediate tracking (useful for testing)
- **`max_age`** (default: 30): Maximum frames to keep track alive without detection
- **`iou_threshold`** (default: 0.3): Minimum IoU for associating detection to existing track
- **`asso_func`**: Association function ("iou", "giou", "ciou", "diou", "ct_dist")

---

## Notes for Debugging

1. **Empty output on first frames**: This is normal if `min_hits > 1`. OC-Sort waits for confirmation.

2. **Shape mismatches**: Check that input to `ocsort.update()` has exactly 6 columns.

3. **Wrong tracker IDs**: Ensure boxes are in absolute pixel coordinates, not normalized.

4. **Deprecation warning**: If you see `torch.range()` warnings, this confirms the bug.

5. **Type errors**: Make sure `class_id` and `tracker_id` are converted to int when creating `sv.Detections`.

---

## Dependencies

```bash
pip install numpy torch torchvision supervision opencv-python
```

The tests use:
- `libs.ocsort.ocsort.OCSort` - from your local implementation
- `supervision` - for Detections class
- Standard libraries: numpy, torch
