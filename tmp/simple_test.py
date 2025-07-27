#!/usr/bin/env python3
"""
Simple test without numpy/scipy dependencies.
"""

import json

def make_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Test with pure Python types
test_data = {
    'python_bool': True,
    'python_false': False,
    'python_int': 42,
    'python_float': 3.14159,
    'significance_flags': {
        'significant_p01': bool(0.005 < 0.01),
        'significant_p05': bool(0.005 < 0.05),
        'significant_p10': bool(0.005 < 0.10)
    },
    'nested': {
        'inner_bool': bool(True),
        'comparison': bool(0.03 < 0.05)
    }
}

print("🧪 Testing JSON serialization with Python types...")
print(f"Original data: {test_data}")

try:
    # Test direct serialization
    json_str = json.dumps(test_data, indent=2)
    print("✅ Direct JSON serialization successful")
    
    # Test with our helper
    clean_data = make_json_serializable(test_data)
    json_str2 = json.dumps(clean_data, indent=2)
    print("✅ Helper function serialization successful")
    
    # Test round-trip
    parsed = json.loads(json_str2)
    print("✅ Round-trip successful")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 