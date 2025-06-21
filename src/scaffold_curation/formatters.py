"""
Data Format Converters

Handles conversion between different data formats used in MindCube project.
"""

import json
from typing import Dict, List, Any, Optional


def format_cogmap_json(cogmap: Dict[str, Any]) -> str:
    """
    Format a cognitive map dictionary into specific JSON string format.
    
    This function is preserved from the original implementation.
    
    Args:
        cogmap: Dictionary with 'objects' and 'views' keys
        
    Returns:
        Formatted JSON string
    """
    result = "{\n"
    result += '  "objects": [\n'
    for i, obj in enumerate(cogmap["objects"]):
        result += '    ' + json.dumps(obj, ensure_ascii=False)
        if i < len(cogmap["objects"]) - 1:
            result += ','
        result += '\n'
    result += '  ],\n'
    
    result += '  "views": [\n'
    for i, view in enumerate(cogmap["views"]):
        result += '    ' + json.dumps(view, ensure_ascii=False)
        if i < len(cogmap["views"]) - 1:
            result += ','
        result += '\n'
    result += '  ]\n'
    result += '}'
    
    return result


def convert_to_qwen_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert data item to Qwen training format.
    
    Args:
        item: Original data item
        
    Returns:
        Item in Qwen format
    """
    # Basic structure for Qwen format
    qwen_item = {
        "id": item.get("id", ""),
        "conversations": [
            {
                "from": "human",
                "value": item.get("question", "")
            },
            {
                "from": "gpt", 
                "value": item.get("answer", "")
            }
        ]
    }
    
    # Add images if present
    if "images" in item:
        qwen_item["image"] = item["images"]
    
    # Add cognitive map information if present
    if "cogmap" in item:
        qwen_item["cogmap"] = item["cogmap"]
    
    return qwen_item


def convert_to_training_format(item: Dict[str, Any], format_type: str = "full") -> Dict[str, Any]:
    """
    Convert data item to training format.
    
    Args:
        item: Original data item
        format_type: Target format type
        
    Returns:
        Item in training format
    """
    if format_type == "qwen":
        return convert_to_qwen_format(item)
    else:
        # For other formats, return as-is for now
        return item


def validate_cogmap_format(cogmap_str: str) -> bool:
    """
    Validate that cognitive map string is valid JSON.
    
    Args:
        cogmap_str: Cognitive map as JSON string
        
    Returns:
        True if valid JSON format
    """
    try:
        parsed = json.loads(cogmap_str)
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False


def normalize_cogmap_format(cogmap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize cognitive map to standard format.
    
    Args:
        cogmap: Cognitive map dictionary
        
    Returns:
        Normalized cognitive map
    """
    normalized = {
        "objects": [],
        "views": []
    }
    
    # Handle different input formats
    if "objects" in cogmap and "views" in cogmap:
        # Already in correct format
        normalized = cogmap
    else:
        # Convert from simplified format
        for key, value in cogmap.items():
            if isinstance(value, dict) and "position" in value:
                obj_dict = {"name": key, "position": value["position"]}
                if "facing" in value:
                    obj_dict["facing"] = value["facing"]
                normalized["objects"].append(obj_dict)
    
    return normalized 