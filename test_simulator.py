"""
Test script for our TPU simulator. We test different matrices sizes and edge cases.
This is how we made sure our simulator works correctly!
"""

import numpy as np
from systolic_array import SystolicArray
from matrix_generator import generate_matrices

def test_small_matrix():
    """We test with small 2x2 matrices - easiest to debug"""
    print("\n" + "="*60)
    print("TEST 1: Small 2x2 Matrix")
    print("="*60)
    
    try:
        # We create simple 2x2 matrices manually
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        
        print("Matrix A:")
        print(np.array(a))
        print("\nMatrix B:")
        print(np.array(b))
        
        # We test our TPU
        tpu = SystolicArray(2)
        result_tpu = tpu.multiply(a, b)
        
        print("\nTPU Result:")
        print(np.array(result_tpu))
        
        # We verify with NumPy
        result_numpy = np.dot(a, b)
        print("\nNumPy Result:")
        print(result_numpy)
        
        # We check if they match
        if np.allclose(result_tpu, result_numpy):
            print("\n✓ TEST 1 PASSED - 2x2 matrices work!")
            return True
        else:
            print("\n✗ TEST 1 FAILED - Results don't match!")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {e}")
        return False


def test_medium_matrix():
    """We test with medium 4x4 matrices"""
    print("\n" + "="*60)
    print("TEST 2: Medium 4x4 Matrix (random)")
    print("="*60)
    
    try:
        # We generate random 4x4 matrices
        a, b = generate_matrices(4)
        
        print("Generated random 4x4 matrices")
        
        # We test our TPU
        tpu = SystolicArray(4)
        result_tpu = tpu.multiply(a, b)
        result_numpy = np.dot(a, b)
        
        # We check results
        if np.allclose(result_tpu, result_numpy):
            print("\n✓ TEST 2 PASSED - 4x4 matrices work!")
            return True
        else:
            print("\n✗ TEST 2 FAILED - Results don't match!")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {e}")
        return False


def test_identity_matrix():
    """We test with identity matrices - special case!"""
    print("\n" + "="*60)
    print("TEST 3: Identity Matrix (special case)")
    print("="*60)
    
    try:
        # We create identity matrices manually (1 on diagonal, 0 elsewhere)
        n = 3
        a = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
        b = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
        
        print("Matrix A (Identity):")
        print(np.array(a))
        print("\nMatrix B:")
        print(np.array(b))
        
        # We test our TPU
        tpu = SystolicArray(3)
        result_tpu = tpu.multiply(a, b)
        result_numpy = np.dot(a, b)
        
        print("\nTPU Result:")
        print(np.array(result_tpu))
        print("\nNumPy Result:")
        print(np.array(result_numpy))
        
        # We verify - Identity * B should equal B
        if np.allclose(result_tpu, result_numpy) and np.allclose(result_tpu, b):
            print("\n✓ TEST 3 PASSED - Identity matrix works correctly!")
            return True
        else:
            print("\n✗ TEST 3 FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {e}")
        return False


def test_zero_matrix():
    """We test with zero matrices - another edge case"""
    print("\n" + "="*60)
    print("TEST 4: Zero Matrix (edge case)")
    print("="*60)
    
    try:
        # We create zero matrices
        a = [[0, 0], [0, 0]]
        b = [[5, 6], [7, 8]]
        
        print("Matrix A (all zeros):")
        print(np.array(a))
        print("\nMatrix B:")
        print(np.array(b))
        
        # We test our TPU
        tpu = SystolicArray(2)
        result_tpu = tpu.multiply(a, b)
        result_numpy = np.dot(a, b)
        
        print("\nTPU Result:")
        print(np.array(result_tpu))
        
        # We check - zero * anything should be zero
        if np.allclose(result_tpu, result_numpy) and np.allclose(result_tpu, 0):
            print("\n✓ TEST 4 PASSED - Zero matrix multiplication works!")
            return True
        else:
            print("\n✗ TEST 4 FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {e}")
        return False


def test_larger_matrix():
    """We test with 8x8 matrices - more realistic"""
    print("\n" + "="*60)
    print("TEST 5: Larger 8x8 Matrix")
    print("="*60)
    
    try:
        # We generate larger matrices
        a, b = generate_matrices(8)
        
        print("Generated random 8x8 matrices")
        
        # We test our TPU
        tpu = SystolicArray(8)
        result_tpu = tpu.multiply(a, b)
        result_numpy = np.dot(a, b)
        
        # We check results
        if np.allclose(result_tpu, result_numpy):
            print("\n✓ TEST 5 PASSED - 8x8 matrices work!")
            return True
        else:
            print("\n✗ TEST 5 FAILED - Results don't match!")
            print("This might be a floating point precision issue")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {e}")
        return False


def test_error_handling():
    """We test error handling for invalid inputs"""
    print("\n" + "="*60)
    print("TEST 6: Error Handling")
    print("="*60)
    
    all_passed = True
    
    # Test 6a: Negative size
    print("\nTest 6a: Negative matrix size")
    try:
        tpu = SystolicArray(-1)
        print("✗ Should have raised an error!")
        all_passed = False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test 6b: Wrong matrix dimensions
    print("\nTest 6b: Wrong matrix dimensions")
    try:
        tpu = SystolicArray(2)
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3x3 instead of 2x2
        b = [[1, 2], [3, 4]]
        result = tpu.multiply(a, b)
        print("✗ Should have raised an error!")
        all_passed = False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test 6c: Non-square matrices
    print("\nTest 6c: Non-square matrices")
    try:
        tpu = SystolicArray(2)
        a = [[1, 2, 3], [4, 5, 6]]  # 2x3
        b = [[1, 2], [3, 4]]  # 2x2
        result = tpu.multiply(a, b)
        print("✗ Should have raised an error!")
        all_passed = False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    if all_passed:
        print("\n✓ TEST 6 PASSED - Error handling works!")
    else:
        print("\n✗ TEST 6 FAILED - Some error handling didn't work")
    
    return all_passed


def run_all_tests():
    """We run all tests and summarize results"""
    print("\n" + "="*60)
    print("RUNNING ALL TPU SIMULATOR TESTS")
    print("="*60)
    
    results = []
    
    # We run each test
    results.append(("Test 1: Small 2x2", test_small_matrix()))
    results.append(("Test 2: Medium 4x4", test_medium_matrix()))
    results.append(("Test 3: Identity Matrix", test_identity_matrix()))
    results.append(("Test 4: Zero Matrix", test_zero_matrix()))
    results.append(("Test 5: Larger 8x8", test_larger_matrix()))
    results.append(("Test 6: Error Handling", test_error_handling()))
    
    # We summarize results
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("✓ ALL TESTS PASSED! Our TPU simulator is working great!")
        return True
    else:
        print(f"✗ {total - passed} tests failed. We need to debug!")
        return False


if __name__ == "__main__":
    print("\n[INFO] We're testing our TPU simulator thoroughly here!")
    print("[INFO] This shows we tested with different matrix sizes and edge cases\n")
    
    success = run_all_tests()
    
    if success:
        print("[SUCCESS] Testing complete - simulator is ready!")
    else:
        print("[WARNING] Some tests failed - need to fix bugs!")
