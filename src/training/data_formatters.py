"""
Data Formatters for Model Training

Converts MindCube general prompt data into model-specific SFT formats.
Supports extensible architecture for adding new model formats.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Literal
from ..utils import load_jsonl, save_json, ensure_dir

ModelType = Literal["qwen2.5vl", "llava", "instructblip"]


class ModelDataFormatter(ABC):
    """Base class for model-specific data formatters."""
    
    def __init__(self, model_name: str):
        """
        Initialize data formatter.
        
        Args:
            model_name: Name of the model for this formatter
        """
        self.model_name = model_name
    
    @abstractmethod
    def format_conversation(self, item: Dict) -> Dict:
        """
        Convert a single prompt item to model-specific format.
        
        Args:
            item: Prompt item with input_prompt, grounded_output, images, etc.
            
        Returns:
            Model-specific formatted conversation
        """
        pass
    
    @abstractmethod
    def get_output_filename(self, input_filename: str) -> str:
        """
        Generate output filename for this model format.
        
        Args:
            input_filename: Original input filename
            
        Returns:
            Model-specific output filename
        """
        pass
    
    def validate_item(self, item: Dict) -> bool:
        """
        Validate that an item has required fields for conversion.
        
        Args:
            item: Input item to validate
            
        Returns:
            True if item is valid for conversion
        """
        required_fields = ["input_prompt", "grounded_output", "images"]
        
        for field in required_fields:
            if field not in item:
                return False
        
        # Check that images is a list
        if not isinstance(item["images"], list):
            return False
            
        return True
    
    def convert_data(self, prompt_data: List[Dict]) -> List[Dict]:
        """
        Convert a list of prompt items to model format.
        
        Args:
            prompt_data: List of prompt items
            
        Returns:
            List of model-formatted conversations
        """
        converted_data = []
        error_count = 0
        
        for i, item in enumerate(prompt_data):
            try:
                if not self.validate_item(item):
                    print(f"⚠️ Skipping item {i}: Missing required fields")
                    error_count += 1
                    continue
                
                formatted_item = self.format_conversation(item)
                converted_data.append(formatted_item)
                
            except Exception as e:
                print(f"⚠️ Error converting item {i}: {e}")
                error_count += 1
                continue
        
        print(f"📊 Conversion summary: {len(converted_data)} successful, {error_count} errors")
        return converted_data


class QwenDataFormatter(ModelDataFormatter):
    """Data formatter for Qwen2.5-VL model training format."""
    
    def __init__(self):
        super().__init__("qwen2.5vl")
    
    def format_conversation(self, item: Dict) -> Dict:
        """
        Convert prompt item to Qwen conversation format.
        
        Qwen format:
        {
            "images": ["path1.jpg", "path2.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\n<image>\nQuestion..."},
                {"from": "gpt", "value": "Answer..."}
            ]
        }
        """
        images = item["images"]
        input_prompt = item["input_prompt"]
        grounded_output = item["grounded_output"]
        
        # Generate image placeholders
        image_placeholders = "\n".join(["<image>" for _ in images])
        
        # Combine image placeholders with input prompt
        human_value = f"{image_placeholders}\n{input_prompt}"
        
        conversation = {
            "images": images,
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt", 
                    "value": grounded_output
                }
            ]
        }
        
        return conversation
    
    def get_output_filename(self, input_filename: str) -> str:
        """Generate Qwen-specific output filename."""
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_qwen_sft.json"


class LlavaDataFormatter(ModelDataFormatter):
    """Data formatter for LLaVA model training format."""
    
    def __init__(self):
        super().__init__("llava")
    
    def format_conversation(self, item: Dict) -> Dict:
        """
        Convert prompt item to LLaVA conversation format.
        
        LLaVA format (example):
        {
            "id": "unique_id",
            "image": "path_to_image.jpg",  # Usually single image
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion..."},
                {"from": "gpt", "value": "Answer..."}
            ]
        }
        """
        images = item["images"]
        input_prompt = item["input_prompt"]
        grounded_output = item["grounded_output"]
        item_id = item.get("id", "unknown")
        
        # LLaVA typically uses single image, take first one
        # If multiple images, we could create multiple entries or concatenate
        main_image = images[0] if images else ""
        
        # Simple approach: use <image> placeholder for first image
        human_value = f"<image>\n{input_prompt}"
        
        conversation = {
            "id": item_id,
            "image": main_image,
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": grounded_output
                }
            ]
        }
        
        # If multiple images, add them to a separate field
        if len(images) > 1:
            conversation["additional_images"] = images[1:]
        
        return conversation
    
    def get_output_filename(self, input_filename: str) -> str:
        """Generate LLaVA-specific output filename."""
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_llava_sft.json"


class InstructBLIPDataFormatter(ModelDataFormatter):
    """Data formatter for InstructBLIP model training format."""
    
    def __init__(self):
        super().__init__("instructblip")
    
    def format_conversation(self, item: Dict) -> Dict:
        """
        Convert prompt item to InstructBLIP format.
        
        InstructBLIP format (example):
        {
            "image": "path_to_image.jpg",
            "text_input": "Question...",
            "text_output": "Answer..."
        }
        """
        images = item["images"]
        input_prompt = item["input_prompt"]
        grounded_output = item["grounded_output"]
        
        # InstructBLIP typically uses single image
        main_image = images[0] if images else ""
        
        conversation = {
            "image": main_image,
            "text_input": input_prompt,
            "text_output": grounded_output
        }
        
        # If multiple images, add them to a separate field
        if len(images) > 1:
            conversation["additional_images"] = images[1:]
        
        return conversation
    
    def get_output_filename(self, input_filename: str) -> str:
        """Generate InstructBLIP-specific output filename."""
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_instructblip_sft.json"


# Formatter registry
FORMATTER_REGISTRY = {
    "qwen2.5vl": QwenDataFormatter(),
    "llava": LlavaDataFormatter(),
    "instructblip": InstructBLIPDataFormatter()
}


def get_formatter(model_type: ModelType) -> ModelDataFormatter:
    """
    Get a data formatter for the specified model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Data formatter instance
    """
    if model_type not in FORMATTER_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(FORMATTER_REGISTRY.keys())}")
    
    return FORMATTER_REGISTRY[model_type]


def list_supported_models() -> List[str]:
    """Get list of supported model types."""
    return list(FORMATTER_REGISTRY.keys())


def convert_prompts_to_sft_format(input_file: str, output_file: str, 
                                 model_type: ModelType) -> None:
    """
    Convert general prompt data to model-specific SFT format.
    
    Args:
        input_file: Path to input JSONL file with prompt data
        output_file: Path to output JSON file for SFT training
        model_type: Target model type for formatting
    """
    print(f"🔄 Converting prompts to {model_type} SFT format...")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    
    # Load input data
    try:
        prompt_data = load_jsonl(input_file)
        print(f"📊 Loaded {len(prompt_data)} prompt items")
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Get formatter and convert data
    formatter = get_formatter(model_type)
    converted_data = formatter.convert_data(prompt_data)
    
    if not converted_data:
        print("❌ No data was successfully converted")
        return
    
    # Save output
    try:
        ensure_dir(os.path.dirname(output_file))
        save_json(converted_data, output_file)
        print(f"✅ Conversion completed: {output_file}")
        print(f"📊 {len(converted_data)} conversations saved")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")


def batch_convert_prompts_to_sft(input_dir: str, output_dir: str, 
                                model_type: ModelType) -> None:
    """
    Convert all prompt files in a directory to SFT format.
    
    Args:
        input_dir: Directory containing prompt JSONL files
        output_dir: Directory to save SFT JSON files
        model_type: Target model type for formatting
    """
    print(f"📁 Batch converting prompts to {model_type} SFT format...")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all JSONL files
    import glob
    pattern = os.path.join(input_dir, "*.jsonl")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"❌ No JSONL files found in {input_dir}")
        return
    
    print(f"📋 Found {len(input_files)} files to convert")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Get formatter
    formatter = get_formatter(model_type)
    
    # Convert each file
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        output_filename = formatter.get_output_filename(base_name)
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"\n🔧 Converting: {base_name}")
        convert_prompts_to_sft_format(input_file, output_file, model_type)
    
    print(f"\n✅ Batch conversion completed: {output_dir}") 