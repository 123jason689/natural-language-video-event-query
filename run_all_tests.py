#!/usr/bin/env python3
"""
Master test runner for OC-Sort + GDINO integration.

This script runs all tests and provides a comprehensive report.
"""

import sys
import traceback


def run_test(test_name, test_func):
    """Run a single test and report results."""
    print("\n" + "▶ "*30)
    print(f"Running: {test_name}")
    print("▶ "*30)
    
    try:
        test_func()
        print(f"\n✅ {test_name} PASSED")
        return True
    except Exception as e:
        print(f"\n❌ {test_name} FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "OC-SORT + GDINO TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Bug demonstration
    print("\n\n📍 SECTION 1: Bug Demonstration")
    print("-"*80)
    try:
        from unit_test_scripts.test_torch_range_bug import (
            demonstrate_torch_range_bug,
            test_ocsort_with_wrong_shape,
            show_fix_recommendation
        )
        results['Bug Demo'] = run_test(
            "torch.range() Bug Demonstration",
            demonstrate_torch_range_bug
        )
        results['Wrong Shape Test'] = run_test(
            "OC-Sort with Wrong Shape",
            test_ocsort_with_wrong_shape
        )
        show_fix_recommendation()
    except ImportError as e:
        print(f"⚠️  Could not import test_torch_range_bug: {e}")
        results['Bug Tests'] = False
    
    # Test 2: Simple OC-Sort tests
    print("\n\n📍 SECTION 2: Simple OC-Sort Tests")
    print("-"*80)
    try:
        from unit_test_scripts.test_ocsort_simple import (
            test_basic_update,
            test_multi_frame_tracking,
            test_input_validation
        )
        results['Basic Update'] = run_test(
            "Basic OC-Sort Update",
            test_basic_update
        )
        results['Multi-Frame'] = run_test(
            "Multi-Frame Tracking",
            test_multi_frame_tracking
        )
        results['Input Validation'] = run_test(
            "Input Validation",
            test_input_validation
        )
    except ImportError as e:
        print(f"⚠️  Could not import test_ocsort_simple: {e}")
        results['Simple Tests'] = False
    
    # Test 3: Integration tests
    print("\n\n📍 SECTION 3: GDINO Integration Tests")
    print("-"*80)
    try:
        from unit_test_scripts.test_ocsort_gdino_integration import (
            test_format_notes,
            test_ocsort_with_gdino_format,
            test_edge_cases,
            test_tracking_consistency
        )
        test_format_notes()
        results['GDINO Format'] = run_test(
            "OC-Sort with GDINO Format",
            test_ocsort_with_gdino_format
        )
        results['Edge Cases'] = run_test(
            "Edge Cases",
            test_edge_cases
        )
        results['Tracking Consistency'] = run_test(
            "Tracking Consistency",
            test_tracking_consistency
        )
    except ImportError as e:
        print(f"⚠️  Could not import test_ocsort_gdino_integration: {e}")
        results['Integration Tests'] = False
    
    # Summary
    print("\n\n" + "="*80)
    print(" "*25 + "TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
    
    print("-"*80)
    print(f"Total: {total} tests")
    print(f"Passed: {passed} ({100*passed//total if total > 0 else 0}%)")
    print(f"Failed: {failed} ({100*failed//total if total > 0 else 0}%)")
    print("="*80)
    
    # Critical warnings
    print("\n" + "⚠️ "*30)
    print("CRITICAL ISSUES TO ADDRESS:")
    print("⚠️ "*30)
    print("\n1. 🐛 BUG in libs/gdino_process.py line 175:")
    print("   Current: phrase_class_idx = torch.range(0, xyxy.shape[0]).numpy()")
    print("   Fix to:  phrase_class_idx = torch.arange(xyxy.shape[0]).numpy()")
    print("\n2. 📝 This creates shape mismatches and uses deprecated PyTorch API")
    print("\n3. 🔍 Review BUG_REPORT.py for full details")
    print("\n" + "⚠️ "*30)
    
    # Exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
