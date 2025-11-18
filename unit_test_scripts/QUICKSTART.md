# Quick Start Guide: OC-Sort Testing

## 🚀 Quick Run

### Run All Tests
```bash
python run_all_tests.py
```

### Run Individual Tests

**1. Simple OC-Sort Tests** (No dependencies on GDINO)
```bash
python test_ocsort_simple.py
```

**2. Bug Demonstration**
```bash
python test_torch_range_bug.py
```

**3. Full Integration Tests**
```bash
python test_ocsort_gdino_integration.py
```

---

## 📋 What You'll See

### Expected Output Structure

Each test will show:
1. **Input data format** - What's being passed to OC-Sort
2. **Processing steps** - How data is transformed
3. **Output format** - What OC-Sort returns
4. **Validation** - Whether it matches expected format

### Key Things to Check

✅ **Input to OC-Sort.update():**
- Shape: `(N, 6)` where N = number of detections
- Columns: `[x1, y1, x2, y2, score, class_id]`
- Coordinates in absolute pixels (not normalized)

✅ **Output from OC-Sort.update():**
- Shape: `(M, 7)` where M ≤ N (M = tracked objects)
- Columns: `[x1, y1, x2, y2, score, class_id, tracker_id]`
- tracker_id is unique integer per object

❌ **Common Issues:**
- Shape `(N, 7)` input → Wrong! Caused by torch.range() bug
- Empty output in first frames → Normal if min_hits > 1
- Deprecation warning about torch.range() → Confirms the bug

---

## 📊 Test Coverage

| Test File | What It Tests | Run Time |
|-----------|---------------|----------|
| `test_ocsort_simple.py` | Basic OC-Sort functionality | ~1 sec |
| `test_torch_range_bug.py` | Demonstrates the bug | ~1 sec |
| `test_ocsort_gdino_integration.py` | Full GDINO→OC-Sort flow | ~2 sec |
| `run_all_tests.py` | All of the above | ~5 sec |

---

## 🔍 Understanding the Output

### Test Output Legend

- ✅ = Test passed, everything working
- ❌ = Test failed, there's an issue
- ⚠️  = Warning, something to be aware of
- 📝 = Note, informational
- 🐛 = Bug detected

### Normal Behaviors

1. **Empty output in first 1-3 frames**: OC-Sort waits for `min_hits` confirmations
2. **Tracker IDs start from 1**: First detected object gets ID=1
3. **M < N in output**: Some detections filtered out (low confidence, not yet confirmed)

### Abnormal Behaviors

1. **Shape mismatch errors**: Usually the torch.range() bug
2. **All empty outputs**: Check `det_thresh` and `min_hits` values
3. **Tracker IDs changing rapidly**: IoU threshold might be too low

---

## 📦 Dependencies

```bash
pip install numpy torch torchvision supervision opencv-python
```

Already installed in your environment:
- `libs.ocsort.ocsort` (your local implementation)
- `libs.gdino_process` (contains the bug)
- `libs.typings_` (FrameBatch definition)

---

## 🎯 Next Steps

1. **Run the tests**:
   ```bash
   python run_all_tests.py
   ```

2. **Review the output** and check for the bug indicators

3. **Read BUG_REPORT.py** for detailed analysis

4. **Fix the bug** in `libs/gdino_process.py` line 175

5. **Re-run tests** to verify the fix

---

## 💡 Tips

- Start with `test_ocsort_simple.py` to verify basic functionality
- Use `test_torch_range_bug.py` to understand the issue clearly
- Run `test_ocsort_gdino_integration.py` for the full picture
- Check `TEST_README.md` for detailed documentation

---

## 📞 Need Help?

Check these files:
- `TEST_README.md` - Full documentation
- `BUG_REPORT.py` - Detailed bug analysis
- `run_all_tests.py` - Comprehensive test runner

All test files include detailed comments and print statements to help you understand what's happening at each step.
