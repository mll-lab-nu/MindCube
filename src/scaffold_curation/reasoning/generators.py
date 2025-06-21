"""
Reasoning Chain Generators

Unified interface for generating reasoning chains across different spatial reasoning settings.
Provides consistent API and handles conversion of reasoning chains to string format.
"""

import json
import os
from typing import Dict, List, Optional, Union, Literal
from ...utils import ensure_dir

# Import individual reasoning generators
from .rotation_reasoning import generate_rotation_reasoning_chain, add_reasoning_chains_to_data as add_rotation_chains
from .translation_reasoning import generate_reasoning_chain as generate_translation_chain, add_reasoning_chains_to_data as add_translation_chains  
from .among_reasoning import generate_reasoning_chain_among, add_reasoning_chains_to_among_data as add_among_chains
from .around_reasoning import generate_around_reasoning_chain, add_reasoning_chains_to_around_data as add_around_chains

ReasoningType = Literal["around", "among", "translation", "rotation"]


class ReasoningChainGenerator:
    """
    Unified generator for reasoning chains across different spatial reasoning settings.
    
    Provides consistent interface for generating reasoning chains and handles
    conversion to string format as required.
    """
    
    def __init__(self, setting: Optional[ReasoningType] = None):
        """
        Initialize reasoning chain generator.
        
        Args:
            setting: Specific reasoning setting to use (optional)
        """
        self.setting = setting
        self.supported_settings = ["around", "among", "translation", "rotation"]
        
        # Setting detection patterns
        self.setting_patterns = {
            "around": ["around"],
            "among": ["among"],
            "translation": ["translation", "linear"],
            "rotation": ["rotation"]
        }
    
    def detect_setting(self, item_id: str) -> Optional[ReasoningType]:
        """
        Detect reasoning setting from item ID.
        
        Args:
            item_id: Item identifier string
            
        Returns:
            Detected setting or None if not found
        """
        item_id_lower = item_id.lower()
        
        for setting, patterns in self.setting_patterns.items():
            if any(pattern in item_id_lower for pattern in patterns):
                return setting  # type: ignore
        
        return None
    
    def generate_reasoning_chain(self, item: Dict) -> Union[List[str], str]:
        """
        Generate reasoning chain for an item.
        
        Args:
            item: QA data item
            
        Returns:
            Generated reasoning chain (as list or string)
        """
        # Detect setting if not specified
        setting = self.setting
        if not setting:
            setting = self.detect_setting(item.get("id", ""))
        
        if not setting:
            return "Unable to determine reasoning setting for this item."
        
        # Generate reasoning chain based on setting
        try:
            if setting == "rotation":
                return generate_rotation_reasoning(item)
            elif setting == "translation":
                return generate_translation_reasoning(item)
            elif setting == "among":
                return generate_among_reasoning(item)
            elif setting == "around":
                return generate_around_reasoning(item)
            else:
                return f"Reasoning generation for setting '{setting}' not implemented."
        except Exception as e:
            return f"Error generating reasoning chain: {str(e)}"
    
    def add_reasoning_to_item(self, item: Dict) -> Dict:
        """
        Add reasoning chain to an item.
        
        Args:
            item: QA data item
            
        Returns:
            Item with reasoning_chain field added
        """
        reasoning_chain = self.generate_reasoning_chain(item)
        
        # Convert to string format as required
        reasoning_string = convert_reasoning_to_string(reasoning_chain)
        
        # Add to item
        item["reasoning_chain"] = reasoning_string
        
        return item


def generate_around_reasoning(item: Dict) -> str:
    """
    Generate reasoning chain for around setting.
    
    Args:
        item: QA data item
        
    Returns:
        Generated reasoning chain as string
    """
    # Use the around_reasoning module
    try:
        reasoning_chain = generate_around_reasoning_chain(item)
        return convert_reasoning_to_string(reasoning_chain)
    except Exception as e:
        return f"Error in around reasoning generation: {str(e)}"


def generate_among_reasoning(item: Dict) -> str:
    """
    Generate reasoning chain for among setting.
    
    Args:
        item: QA data item
        
    Returns:
        Generated reasoning chain as string
    """
    try:
        reasoning_chain = generate_reasoning_chain_among(item)
        return convert_reasoning_to_string(reasoning_chain)
    except Exception as e:
        return f"Error in among reasoning generation: {str(e)}"


def generate_translation_reasoning(item: Dict) -> str:
    """
    Generate reasoning chain for translation setting.
    
    Args:
        item: QA data item
        
    Returns:
        Generated reasoning chain as string
    """
    try:
        question = item.get('question', '')
        answer = item.get('gt_answer', '')
        category = item.get('category', [])
        meta_info = item.get('meta_info', [])
        question_type = item.get('type', None)
        
        reasoning_chain = generate_translation_chain(
            question=question,
            answer=answer,
            category=category,
            meta_info=meta_info,
            question_type=question_type
        )
        
        return convert_reasoning_to_string(reasoning_chain)
    except Exception as e:
        return f"Error in translation reasoning generation: {str(e)}"


def generate_rotation_reasoning(item: Dict) -> str:
    """
    Generate reasoning chain for rotation setting.
    
    Args:
        item: QA data item
        
    Returns:
        Generated reasoning chain as string
    """
    try:
        reasoning_chain = generate_rotation_reasoning_chain(item)
        return convert_reasoning_to_string(reasoning_chain)
    except Exception as e:
        return f"Error in rotation reasoning generation: {str(e)}"


def convert_reasoning_to_string(reasoning: Union[List[str], str]) -> str:
    """
    Convert reasoning chain to string format as required.
    
    Args:
        reasoning: Reasoning chain as list of strings or single string
        
    Returns:
        Reasoning chain as single string
    """
    if isinstance(reasoning, list):
        # Join list elements with newlines or spaces
        return " ".join(reasoning) if reasoning else ""
    elif isinstance(reasoning, str):
        return reasoning
    else:
        return str(reasoning)


def add_reasoning_chain_to_item(item: Dict, setting: Optional[ReasoningType] = None) -> Dict:
    """
    Add reasoning chain to a single item.
    
    Args:
        item: QA data item
        setting: Specific reasoning setting (optional, will be auto-detected)
        
    Returns:
        Item with reasoning_chain field added
    """
    generator = ReasoningChainGenerator(setting)
    return generator.add_reasoning_to_item(item)


def batch_generate_reasoning_chains(input_file: str, output_file: str, 
                                  setting: Optional[ReasoningType] = None) -> None:
    """
    Generate reasoning chains for all items in a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        setting: Specific reasoning setting (optional, will process all settings)
    """
    print(f"🧠 Generating reasoning chains: {input_file} -> {output_file}")
    
    # Create output directory if needed
    ensure_dir(os.path.dirname(output_file))
    
    # Load data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Filter items by setting if specified
    target_items = []
    if setting:
        generator = ReasoningChainGenerator(setting)
        for item in items:
            detected_setting = generator.detect_setting(item.get("id", ""))
            if detected_setting == setting:
                target_items.append(item)
    else:
        target_items = items
    
    print(f"Processing {len(target_items)} items...")
    
    # Process items
    processed_items = []
    setting_counts = {"around": 0, "among": 0, "translation": 0, "rotation": 0}
    
    for item in target_items:
        try:
            processed_item = add_reasoning_chain_to_item(item.copy(), setting)
            processed_items.append(processed_item)
            
            # Count by setting
            item_setting = ReasoningChainGenerator().detect_setting(item.get("id", ""))
            if item_setting in setting_counts:
                setting_counts[item_setting] += 1
        except Exception as e:
            print(f"⚠️ Error processing item {item.get('id', 'unknown')}: {e}")
            processed_items.append(item)  # Keep original item
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully generated reasoning chains: {output_file}")
        print(f"📊 Processing summary:")
        for setting_name, count in setting_counts.items():
            if count > 0:
                print(f"  - {setting_name}: {count} items")
        
    except Exception as e:
        print(f"❌ Error saving output file: {e}") 