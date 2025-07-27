#!/usr/bin/env python3
"""
Test script to verify JSON serialization works correctly.
"""

import json
import sys
import os

# Add project root to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Try to import scipy
try:
    import scipy.stats
    from scipy.stats import t
    import numpy as np
    SCIPY_AVAILABLE = True
    print("✅ scipy available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available")

# Import our helper function
try:
    from tmp.meta_evaluation import make_json_serializable
    print("✅ Helper function imported")
except ImportError as e:
    print(f"❌ Cannot import helper function: {e}")
    sys.exit(1)

def test_json_serialization():
    """Test different data types for JSON serialization."""
    print("\n🧪 Testing JSON serialization...")
    
    # Test data with potential problematic types
    test_data = {
        'python_bool': True,
        'python_false': False,
        'python_int': 42,
        'python_float': 3.14159,
        'python_str': "test",
        'python_list': [1, 2, 3],
        'nested_dict': {
            'inner_bool': True,
            'inner_float': 2.718
        }
    }
    
    # Add scipy/numpy types if available
    if SCIPY_AVAILABLE:
        print("  Adding scipy/numpy types...")
        test_data.update({
            'numpy_bool_true': np.bool_(True),
            'numpy_bool_false': np.bool_(False),
            'numpy_int': np.int64(123),
            'numpy_float': np.float64(1.414),
            'comparison_result': np.float64(0.05) < 0.01,  # This creates numpy.bool_
            'scipy_available': SCIPY_AVAILABLE
        })
    
    print(f"  Original data types:")
    for key, value in test_data.items():
        print(f"    {key}: {type(value)} = {value}")
    
    # Test without conversion (should fail with scipy types)
    print("\n  Testing raw JSON serialization...")
    try:
        json_str = json.dumps(test_data)
        print("  ✅ Raw serialization successful")
    except TypeError as e:
        print(f"  ❌ Raw serialization failed: {e}")
        print("  This is expected when scipy/numpy types are present")
    
    # Test with our conversion function
    print("\n  Testing with make_json_serializable...")
    try:
        clean_data = make_json_serializable(test_data)
        json_str = json.dumps(clean_data, indent=2)
        print("  ✅ Conversion + serialization successful")
        
        # Verify types after conversion
        print(f"  Converted data types:")
        for key, value in clean_data.items():
            print(f"    {key}: {type(value)} = {value}")
        
        # Test round-trip
        parsed_data = json.loads(json_str)
        print("  ✅ Round-trip (serialize + parse) successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False

def test_statistical_values():
    """Test statistical computation results."""
    print("\n🔬 Testing statistical computation serialization...")
    
    if not SCIPY_AVAILABLE:
        print("  ⚠️  Skipping (scipy not available)")
        return True
    
    # Simulate statistical computation that might produce numpy types
    import numpy as np
    from scipy import stats
    
    # Sample data
    values = [0.75, 0.73, 0.77]
    
    # Statistical computations
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    
    # t-test (produces numpy types)
    t_stat, p_value = stats.ttest_1samp(values, 0)
    
    # Confidence interval
    ci = stats.t.interval(0.95, len(values)-1, mean_val, stats.sem(values))
    
    # Create test result structure
    stats_result = {
        'mean': mean_val,
        'std': std_val,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'significant_p05': p_value < 0.05,
        'significant_p01': p_value < 0.01,
    }
    
    print(f"  Original statistical types:")
    for key, value in stats_result.items():
        print(f"    {key}: {type(value)} = {value}")
    
    # Test serialization
    try:
        clean_stats = make_json_serializable(stats_result)
        json_str = json.dumps(clean_stats, indent=2)
        print("  ✅ Statistical data serialization successful")
        
        print(f"  Converted statistical types:")
        for key, value in clean_stats.items():
            print(f"    {key}: {type(value)} = {value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Statistical data serialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 JSON Serialization Test Suite")
    print("=" * 40)
    
    success1 = test_json_serialization()
    success2 = test_statistical_values()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("✅ All tests passed! JSON serialization should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 