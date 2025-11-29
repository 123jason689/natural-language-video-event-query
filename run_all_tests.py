#!/usr/bin/env python3
"""
Master test runner for OC-Sort + GDINO + MobileViCLIP integration.

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
        success = test_func()
        if success is None or type(success) is not bool: # Handle functions that don't return bool but raise assert errors
            success = True
        
        if success:
            print(f"\n✅ {test_name} PASSED")
            return True
        else:
            print(f"\n❌ {test_name} FAILED (Function returned False)")
            return False
    except Exception as e:
        print(f"\n❌ {test_name} FAILED (Exception)")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "VIDEO PIPELINE TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Simple OC-Sort tests
    print("\n\n📍 SECTION 1: Simple OC-Sort Tests")
    print("-"*80)
    try:
        from unit_test_scripts.test_ocsort_simple import (
            test_basic_update,
            test_multi_frame_tracking,
            test_multi_object_tracking,
            test_input_validation
        )
        results['Basic Update'] = run_test(
            "Basic OC-Sort Update",
            test_basic_update
        )
        results['Single Object Multi-Frame'] = run_test(
            "Single Object Multi-Frame Tracking",
            test_multi_frame_tracking
        )
        results['Multi-Object Multi-Frame'] = run_test(
            "Multi-Object Multi-Frame Tracking",
            test_multi_object_tracking
        )
        results['Input Validation'] = run_test(
            "Input Validation",
            test_input_validation
        )
    except ImportError as e:
        print(f"⚠️  Could not import test_ocsort_simple: {e}")
        results['Simple Tests'] = False
    
    # Test 2: Integration tests
    print("\n\n📍 SECTION 2: GDINO Integration Tests")
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

    # Test 3: MobileViCLIP Preprocessing
    print("\n\n📍 SECTION 3: MobileViCLIP Preprocessing Tests")
    print("-"*80)
    try:
        from unit_test_scripts.test_mobileviclip_preprocessing import (
            test_preprocessing_end_to_end
        )
        results['MobileViCLIP Preprocessing'] = run_test(
            "MobileViCLIP Pipeline",
            test_preprocessing_end_to_end
        )
    except ImportError as e:
        print(f"⚠️  Could not import test_mobileviclip_preprocessing: {e}")
        results['MobileViCLIP Tests'] = False
    
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
    
    if failed > 0:
        print("\n" + "⚠️ "*30)
        print("PLEASE REVIEW FAILED TESTS ABOVE")
        print("⚠️ "*30)
    
    # Exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())