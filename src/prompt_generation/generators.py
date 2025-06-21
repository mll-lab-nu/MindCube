"""
Prompt Generators

Core prompt generation logic for different task settings.
Each generator handles the creation of input_prompt and grounded_output for specific tasks.
"""

import json
import os
from typing import Dict, List, Optional, Union, Literal
from .templates import get_template, list_templates

TaskType = Literal[
    "raw_qa", "ff_rsn", 
    "aug_cgmap_in", "aug_cgmap_out", "plain_cgmap_out", 
    "plain_cgmap_ffr_out", "aug_cgmap_ffr_out", "cgmap_in_ffr_out"
]


class PromptGenerator:
    """Base class for prompt generators."""
    
    def __init__(self, task_type: TaskType):
        """
        Initialize prompt generator.
        
        Args:
            task_type: Type of task for prompt generation
        """
        self.task_type = task_type
        self.template = get_template(task_type)
    
    def generate_item(self, item: Dict) -> Dict:
        """
        Generate input_prompt and grounded_output for a single item.
        
        Args:
            item: Input data item with scaffold data
            
        Returns:
            Item with all necessary fields plus input_prompt and grounded_output
        """
        # Create a copy to preserve all original data
        result_item = item.copy()
        
        # Generate and add new fields
        result_item["input_prompt"] = self.template.generate_prompt(item)
        result_item["grounded_output"] = self.template.generate_output(item)
        
        return result_item
    
    def validate_item(self, item: Dict) -> bool:
        """
        Validate that an item has required fields for this generator.
        
        Args:
            item: Data item to validate
            
        Returns:
            True if item is valid for this generator
        """
        required_fields = ["id", "question", "gt_answer"]
        
        for field in required_fields:
            if field not in item:
                return False
        
        return True
    
    def get_required_fields(self) -> List[str]:
        """Get list of required fields for this generator."""
        return ["id", "question", "gt_answer"]


class RawQAGenerator(PromptGenerator):
    """Generator for raw QA tasks without cognitive maps or reasoning."""
    
    def __init__(self):
        super().__init__("raw_qa")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for raw QA generation."""
        # Raw QA only needs basic fields
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "images"]

class FFRSNGenerator(PromptGenerator):
    """Generator for FF-RSN tasks."""
    
    def __init__(self):
        super().__init__("ff_rsn")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for FF-RSN generation."""
        # FF-RSN needs basic fields plus reasoning_chain
        if not super().validate_item(item):
            return False
        return "reasoning_chain" in item
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "reasoning_chain"]


class AugCGMapInGenerator(PromptGenerator):
    """Generator for Aug-CGMap-In tasks."""
    
    def __init__(self):
        super().__init__("aug_cgmap_in")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for Aug-CGMap-In generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "grounded_cogmap"]


class AugCGMapOutGenerator(PromptGenerator):
    """Generator for Aug-CGMap-Out tasks."""
    
    def __init__(self):
        super().__init__("aug_cgmap_out")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for Aug-CGMap-Out generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "grounded_cogmap"]


class PlainCGMapOutGenerator(PromptGenerator):
    """Generator for Plain-CGMap-Out tasks."""
    
    def __init__(self):
        super().__init__("plain_cgmap_out")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for Plain-CGMap-Out generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "plain_cogmap"]


class PlainCGMapFFROutGenerator(PromptGenerator):
    """Generator for Plain-CGMap-FFR-Out tasks."""
    
    def __init__(self):
        super().__init__("plain_cgmap_ffr_out")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for Plain-CGMap-FFR-Out generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "plain_cogmap", "reasoning_chain"]


class AugCGMapFFROutGenerator(PromptGenerator):
    """Generator for Aug-CGMap-FFR-Out tasks."""
    
    def __init__(self):
        super().__init__("aug_cgmap_ffr_out")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for Aug-CGMap-FFR-Out generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "grounded_cogmap", "reasoning_chain"]


class CGMapInFFROutGenerator(PromptGenerator):
    """Generator for CGMap-In-FFR-Out tasks."""
    
    def __init__(self):
        super().__init__("cgmap_in_ffr_out")
    
    def validate_item(self, item: Dict) -> bool:
        """Validate item for CGMap-In-FFR-Out generation."""
        # TODO: Implement validation logic
        return super().validate_item(item)
    
    def get_required_fields(self) -> List[str]:
        return ["id", "question", "gt_answer", "grounded_cogmap", "reasoning_chain"]


# Generator registry
GENERATOR_REGISTRY = {
    "raw_qa": RawQAGenerator(),
    "ff_rsn": FFRSNGenerator(),
    "aug_cgmap_in": AugCGMapInGenerator(),
    "aug_cgmap_out": AugCGMapOutGenerator(),
    "plain_cgmap_out": PlainCGMapOutGenerator(),
    "plain_cgmap_ffr_out": PlainCGMapFFROutGenerator(),
    "aug_cgmap_ffr_out": AugCGMapFFROutGenerator(),
    "cgmap_in_ffr_out": CGMapInFFROutGenerator(),
}


def get_generator(task_type: TaskType) -> PromptGenerator:
    """
    Get a prompt generator for the specified task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Prompt generator instance
    """
    if task_type not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(GENERATOR_REGISTRY.keys())}")
    
    return GENERATOR_REGISTRY[task_type]


def list_task_types() -> List[str]:
    """List all available task types."""
    return list(GENERATOR_REGISTRY.keys())


def auto_detect_task_type(item: Dict) -> Optional[TaskType]:
    """
    Auto-detect the best task type for an item based on available fields.
    
    Args:
        item: Data item to analyze
        
    Returns:
        Detected task type or None if no suitable type found
    """
    # Check basic requirements first
    if "question" not in item or "gt_answer" not in item:
        return None
    
    # Check for reasoning_chain to detect ff_rsn
    if "reasoning_chain" in item:
        return "ff_rsn"
    else:
        return "raw_qa"


def generate_prompts_for_item(item: Dict, task_type: Optional[TaskType] = None) -> Dict:
    """
    Generate prompts for a single item.
    
    Args:
        item: Input data item
        task_type: Specific task type (auto-detected if None)
        
    Returns:
        Item with input_prompt and grounded_output fields added
    """
    # Auto-detect task type if not specified
    if task_type is None:
        task_type = auto_detect_task_type(item)
        if task_type is None:
            raise ValueError(f"Cannot determine task type for item {item.get('id', 'unknown')}")
    
    # Get appropriate generator
    generator = get_generator(task_type)
    
    # Validate item
    if not generator.validate_item(item):
        raise ValueError(f"Item {item.get('id', 'unknown')} is not valid for task type {task_type}")
    
    # Generate prompts
    return generator.generate_item(item)


def batch_generate_prompts_for_items(items: List[Dict], task_type: Optional[TaskType] = None) -> List[Dict]:
    """
    Generate prompts for a list of items.
    
    Args:
        items: List of input data items
        task_type: Specific task type (auto-detected if None)
        
    Returns:
        List of items with prompts generated
    """
    results = []
    
    for item in items:
        try:
            result_item = generate_prompts_for_item(item, task_type)
            results.append(result_item)
        except Exception as e:
            print(f"⚠️ Error processing item {item.get('id', 'unknown')}: {e}")
            # Keep original item on error
            results.append(item)
    
    return results 