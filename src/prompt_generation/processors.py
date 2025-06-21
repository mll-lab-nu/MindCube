"""
Prompt Processing

File processing, validation, and batch operations for prompt generation.
Handles JSONL files, directory organization, and result formatting.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..utils import load_jsonl, save_jsonl, ensure_dir
from .generators import (
    TaskType, 
    get_generator,
    list_task_types, 
    auto_detect_task_type,
    generate_prompts_for_item,
    batch_generate_prompts_for_items
)


def get_default_prompt_output_dir() -> str:
    """
    Get default output directory for prompt generation.
    All prompts go to ./data/prompts/general/ by default.
    
    Returns:
        Default prompt output directory path
    """
    return "./data/prompts/general"


def generate_task_prompts(input_file: str, output_file: str, 
                         task_type: Optional[TaskType] = None,
                         auto_detect: bool = True) -> None:
    """
    Generate prompts for a specific task from scaffold data.
    
    Output contains all input fields plus input_prompt and grounded_output.
    All scaffold fields (cogmap, reasoning_chain, grounded_cogmap, etc.) are preserved.
    
    Args:
        input_file: Path to input JSONL file with scaffold data
        output_file: Path to output JSONL file for prompts
        task_type: Specific task type (auto-detected if None and auto_detect=True)
        auto_detect: Whether to auto-detect task type for each item
    """
    print(f"💬 Generating prompts: {input_file} -> {output_file}")
    if task_type:
        print(f"🎯 Task type: {task_type}")
    elif auto_detect:
        print(f"🔍 Auto-detecting task types")
    
    # Create output directory
    ensure_dir(os.path.dirname(output_file))
    
    # Load input data
    try:
        items = load_jsonl(input_file)
        print(f"📁 Loaded {len(items)} items from {input_file}")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Process items
    processed_items = []
    task_counts = {
        "raw_qa": 0, "ff_rsn": 0, 
        "aug_cgmap_in": 0, "aug_cgmap_out": 0, "plain_cgmap_out": 0,
        "plain_cgmap_ffr_out": 0, "aug_cgmap_ffr_out": 0, "cgmap_in_ffr_out": 0,
        "cogmap_qa": 0, "reasoning_qa": 0, "full_pipeline": 0
    }
    error_count = 0
    
    for item in items:
        try:
            if auto_detect and task_type is None:
                # Auto-detect task type for each item
                item_task_type = auto_detect_task_type(item)
                if item_task_type is None:
                    print(f"⚠️ Cannot detect task type for item {item.get('id', 'unknown')}")
                    processed_items.append(item)  # Keep original
                    error_count += 1
                    continue
            else:
                item_task_type = task_type
            
            # Generate prompts
            processed_item = generate_prompts_for_item(item, item_task_type)
            processed_items.append(processed_item)
            
            # Count by task type
            if item_task_type in task_counts:
                task_counts[item_task_type] += 1
                
        except Exception as e:
            print(f"⚠️ Error processing item {item.get('id', 'unknown')}: {e}")
            processed_items.append(item)  # Keep original
            error_count += 1
    
    # Save results
    try:
        save_jsonl(processed_items, output_file)
        print(f"✅ Prompt generation completed: {output_file}")
        
        # Print statistics
        print(f"📊 Processing summary:")
        total_processed = sum(task_counts.values())
        for task_name, count in task_counts.items():
            if count > 0:
                print(f"  - {task_name}: {count} items")
        
        if error_count > 0:
            print(f"  - errors: {error_count} items")
        
        print(f"  - total: {total_processed} successfully processed, {error_count} errors")
        
    except Exception as e:
        print(f"❌ Error saving output file: {e}")


def generate_all_task_prompts(input_file: str, output_dir: Optional[str] = None) -> None:
    """
    Generate prompts for ALL task types from a single input file.
    All task types get saved to the same directory (default: ./data/prompts/general/).
    
    Output naming: <input_basename>_<task_name>.jsonl
    
    Args:
        input_file: Path to input JSONL file with scaffold data
        output_dir: Custom output directory (uses default if None)
    """
    from pathlib import Path
    
    input_path = Path(input_file)
    base_name = input_path.stem  # filename without extension
    
    print(f"🔄 Generating ALL task prompts from: {input_file}")
    
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = get_default_prompt_output_dir()
    
    print(f"📁 Output directory: {output_dir}")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Get all available task types
    from typing import cast
    all_task_types = cast(List[TaskType], list_task_types())
    
    print(f"🎯 Task types to generate: {len(all_task_types)}")
    for task_type in all_task_types:
        print(f"  - {task_type}")
    
    # Generate each task type
    for task_type in all_task_types:
        output_file = os.path.join(output_dir, f"{base_name}_{task_type}.jsonl")
        
        print(f"\n🔧 Processing {task_type}...")
        print(f"  📁 Saving to: {output_file}")
        
        try:
            generate_task_prompts(input_file, output_file, task_type, auto_detect=False)
        except Exception as e:
            print(f"❌ Error processing {task_type}: {e}")
            continue
    
    print(f"\n✅ All task prompt generation completed!")
    print(f"📁 Results saved to: {output_dir}")


def batch_generate_prompts(input_dir: str, output_dir: str,
                          task_types: Optional[List[TaskType]] = None,
                          auto_detect: bool = True) -> None:
    """
    Generate prompts for multiple files and/or multiple task types.
    
    Args:
        input_dir: Directory containing input JSONL files
        output_dir: Directory to save prompt files
        task_types: List of task types to generate (generates all if None)
        auto_detect: Whether to auto-detect task types
    """
    print(f"📁 Batch prompt generation: {input_dir} -> {output_dir}")
    
    # Find input files
    pattern = os.path.join(input_dir, "*.jsonl")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"❌ No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Determine task types to process
    use_auto_detect = task_types is None and auto_detect
    if task_types is None:
        if auto_detect:
            # Auto-detect mode - no specific task types needed
            actual_task_types: List[TaskType] = []
        else:
            # Use all available task types
            from typing import cast
            actual_task_types = cast(List[TaskType], list_task_types())
    else:
        actual_task_types = task_types
    
    if use_auto_detect:
        print(f"Task types: auto-detect")
    else:
        print(f"Task types: {actual_task_types}")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Process each file
    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        if use_auto_detect:
            # Auto-detect mode: one output file per input file
            output_file = os.path.join(output_dir, f"{base_name}_prompts.jsonl")
            print(f"\n🔧 Processing {base_name} (auto-detect)...")
            generate_task_prompts(input_file, output_file, None, auto_detect=True)
        else:
            # Specific task types: one output file per task type
            for task_type in actual_task_types:
                output_file = os.path.join(output_dir, f"{base_name}_{task_type}_prompts.jsonl")
                print(f"\n🔧 Processing {base_name} ({task_type})...")
                generate_task_prompts(input_file, output_file, task_type, auto_detect=False)
    
    print(f"\n✅ Batch prompt generation completed. Results saved to {output_dir}")


class PromptProcessor:
    """
    Processor for prompt generation with validation and statistics.
    """
    
    def __init__(self, task_type: Optional[TaskType] = None):
        """
        Initialize prompt processor.
        
        Args:
            task_type: Specific task type (auto-detected if None)
        """
        self.task_type = task_type
        self.generator = get_generator(task_type) if task_type else None
    
    def process_file(self, input_file: str, output_file: str) -> Dict:
        """
        Process a single file and return statistics.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            
        Returns:
            Processing statistics
        """
        # self.task_type is already TaskType | None, so it's compatible
        generate_task_prompts(input_file, output_file, self.task_type, auto_detect=(self.task_type is None))
        
        return {
            "input_file": input_file,
            "output_file": output_file,
            "task_type": self.task_type or "auto_detect",
            "status": "completed"
        }
    
    def validate_item(self, item: Dict) -> bool:
        """
        Validate that an item can be processed.
        
        Args:
            item: Data item to validate
            
        Returns:
            True if item is valid for processing
        """
        if self.generator:
            return self.generator.validate_item(item)
        else:
            # Auto-detect mode: check if any task type can handle it
            return auto_detect_task_type(item) is not None
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types."""
        return list_task_types()


def validate_scaffold_data(input_file: str) -> Dict:
    """
    Validate scaffold data file and provide statistics about available fields.
    
    Args:
        input_file: Path to scaffold data file
        
    Returns:
        Validation report with statistics
    """
    print(f"🔍 Validating scaffold data: {input_file}")
    
    try:
        items = load_jsonl(input_file)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_items": 0
        }
    
    # Analyze available fields
    field_counts = {}
    task_type_counts = {
        "raw_qa": 0, "ff_rsn": 0, 
        "aug_cgmap_in": 0, "aug_cgmap_out": 0, "plain_cgmap_out": 0,
        "plain_cgmap_ffr_out": 0, "aug_cgmap_ffr_out": 0, "cgmap_in_ffr_out": 0,
        "cogmap_qa": 0, "reasoning_qa": 0, "full_pipeline": 0, "unknown": 0
    }
    
    for item in items:
        # Count field availability
        for field in item.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
        
        # Detect task type
        task_type = auto_detect_task_type(item)
        if task_type:
            task_type_counts[task_type] += 1
        else:
            task_type_counts["unknown"] += 1
    
    total_items = len(items)
    
    # Create report
    report = {
        "status": "success",
        "total_items": total_items,
        "field_availability": {
            field: {"count": count, "percentage": count/total_items*100}
            for field, count in field_counts.items()
        },
        "task_type_distribution": task_type_counts,
        "recommendations": []
    }
    
    # Add recommendations
    if task_type_counts["unknown"] > 0:
        report["recommendations"].append(f"{task_type_counts['unknown']} items cannot be processed (missing required fields)")
    
    if field_counts.get("cogmap", 0) > 0:
        report["recommendations"].append("Cognitive maps available - can generate cogmap_qa prompts")
    
    if field_counts.get("reasoning_chain", 0) > 0:
        report["recommendations"].append("Reasoning chains available - can generate reasoning_qa prompts")
    
    if field_counts.get("cogmap", 0) > 0 and field_counts.get("reasoning_chain", 0) > 0:
        report["recommendations"].append("Both cogmap and reasoning available - can generate full_pipeline prompts")
    
    print(f"✅ Validation completed:")
    print(f"  - Total items: {total_items}")
    print(f"  - Task type distribution: {task_type_counts}")
    
    return report


def quick_prompt_sample(input_file: str, task_type: Optional[TaskType] = None, 
                       num_samples: int = 3) -> None:
    """
    Generate a quick sample of prompts for preview.
    
    Args:
        input_file: Path to input file
        task_type: Task type (auto-detected if None)
        num_samples: Number of samples to generate
    """
    print(f"📝 Generating prompt samples from {input_file}")
    
    # Load data
    try:
        items = load_jsonl(input_file)
        sample_items = items[:num_samples]
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Process samples
    for i, item in enumerate(sample_items):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {item.get('id', 'unknown')}")
        
        try:
            processed_item = generate_prompts_for_item(item, task_type)
            detected_task = auto_detect_task_type(item) if task_type is None else task_type
            
            print(f"Detected task type: {detected_task}")
            print(f"Input prompt preview:")
            prompt_preview = processed_item["input_prompt"][:200] + "..." if len(processed_item["input_prompt"]) > 200 else processed_item["input_prompt"]
            print(prompt_preview)
            
            print(f"Grounded output preview:")
            output_preview = processed_item["grounded_output"][:200] + "..." if len(processed_item["grounded_output"]) > 200 else processed_item["grounded_output"]
            print(output_preview)
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n✅ Sample generation completed") 