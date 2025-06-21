"""
Data Processors

Main interface for data processing operations with minimal changes to original code.
Provides unified interfaces for cognitive map generation, reasoning chain generation,
and data format conversion.

Updated to support modular reasoning chain generation with selective processing
and organized directory structure for scaffold data.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Literal
from ..utils import load_jsonl, save_jsonl, ensure_dir
from .cogmap.generators import CogMapGenerator, generate_with_reasoning
from .reasoning.generators import ReasoningChainGenerator, batch_generate_reasoning_chains

TaskType = Literal["cognitive_map", "base_sft", "reasoning", "full_pipeline"]
FormatType = Literal["full", "shortened", "qwen"]
InstructionType = Literal["aug", "plain", "both"]
ReasoningType = Literal["around", "among", "translation", "rotation"]


def get_scaffold_directory_structure(task_type: TaskType) -> str:
    """
    Determine scaffold directory structure based on task type.
    
    Args:
        task_type: Type of scaffold processing task
    
    Returns:
        Directory path following the convention: ./data/scaffold/{category}/
    """
    base_dir = "./data/scaffold"
    
    # Map task types to categories for scaffold organization
    task_categories = {
        "cognitive_map": "cogmap",
        "base_sft": "all", 
        "reasoning": "reasoning",
        "full_pipeline": "all"  # full pipeline goes to 'all' since it combines everything
    }
    
    category = task_categories.get(task_type, "all")
    return f"{base_dir}/{category}"


def get_default_scaffold_output_path(input_path: str, task_type: TaskType) -> str:
    """
    Get default output path for scaffold data with organized directory structure.
    
    Args:
        input_path: Original input file path
        task_type: Type of scaffold processing task
        
    Returns:
        Organized output path: ./data/scaffold/{category}/{filename}
    """
    input_path_obj = Path(input_path)
    filename = input_path_obj.name  # Keep original filename
    
    output_dir = get_scaffold_directory_structure(task_type)
    ensure_dir(output_dir)
    
    return os.path.join(output_dir, filename)


def generate_cognitive_maps(input_path: str, output_path: str, 
                          format_type: FormatType = "full",
                          instruction_type: InstructionType = "both",
                          suppress_warnings: bool = False) -> None:
    """
    Generate cognitive maps for a JSONL file.
    
    This is the main interface that preserves the original generate_cogmap functionality
    with minimal changes.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        format_type: Cognitive map format ("full" or "shortened")
    """
    print(f"🔧 Generating cognitive maps: {input_path} -> {output_path}")
    
    # Create generator
    generator = CogMapGenerator(format_type=format_type, suppress_warnings=suppress_warnings)
    
    # Load data
    try:
        data = load_jsonl(input_path)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Process data maintaining original order
    processed_data = []
    
    # Classify items by setting (preserving original logic)
    settings_data = {
        "around": [],
        "among": [],
        "translation": [],
        "rotation": []
    }
    
    for item in data:
        item_id = item.get("id", "")
        setting = generator.detect_setting(item_id)
        
        if setting:
            settings_data[setting].append(item)
            # Process the item and add to processed_data
            processed_item = generator.add_cogmap_to_item(item.copy())
            processed_data.append(processed_item)
        else:
            print(f"⚠️ Warning: Unknown setting for item {item_id}")
            processed_data.append(item)  # Keep original item
    
    # Print statistics (preserving original behavior)
    total_questions = len(processed_data)
    print(f"Total number of questions: {total_questions}")
    
    settings_counts = [
        f"{len(settings_data['around'])} around questions",
        f"{len(settings_data['among'])} among questions", 
        f"{len(settings_data['translation'])} translation questions",
        f"{len(settings_data['rotation'])} rotation questions"
    ]
    print(f"Found {', '.join(settings_counts)}")
    
    # Save processed data
    try:
        save_jsonl(processed_data, output_path)
        print(f"✅ Successfully generated cognitive maps: {output_path}")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")


def generate_reasoning_chains(input_path: str, output_path: str,
                            format_type: FormatType = "full",
                            reasoning_setting: Optional[ReasoningType] = None) -> None:
    """
    Generate reasoning chains for a JSONL file.
    
    Now supports real reasoning chain generation with modular reasoning modules.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file  
        format_type: Output format type (not used for reasoning, kept for compatibility)
        reasoning_setting: Specific reasoning setting to process (optional)
    """
    print(f"🧠 Generating reasoning chains: {input_path} -> {output_path}")
    if reasoning_setting:
        print(f"🎯 Target setting: {reasoning_setting}")
    else:
        print(f"🎯 Processing all reasoning settings")
    
    # Use the batch processing function from reasoning module
    batch_generate_reasoning_chains(input_path, output_path, reasoning_setting)


def generate_full_pipeline(input_path: str, output_path: str,
                         format_type: FormatType = "full",
                         reasoning_setting: Optional[ReasoningType] = None) -> None:
    """
    Generate both cognitive maps and reasoning chains.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        format_type: Output format type
        reasoning_setting: Specific reasoning setting (optional)
    """
    print(f"🔧🧠 Full pipeline processing: {input_path} -> {output_path}")
    if reasoning_setting:
        print(f"🎯 Target reasoning setting: {reasoning_setting}")
    
    # Load data
    try:
        data = load_jsonl(input_path)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Process with both cognitive maps and reasoning
    processed_data = []
    
    # Create generators
    cogmap_generator = CogMapGenerator(format_type=format_type)
    reasoning_generator = ReasoningChainGenerator(reasoning_setting)
    
    # Statistics tracking
    cogmap_settings_data = {
        "around": [],
        "among": [],
        "translation": [],
        "rotation": []
    }
    reasoning_setting_counts = {"around": 0, "among": 0, "translation": 0, "rotation": 0}
    
    for item in data:
        # Add cognitive map
        processed_item = cogmap_generator.add_cogmap_to_item(item.copy())
        
        # Track cogmap settings
        item_id = item.get("id", "")
        cogmap_setting = cogmap_generator.detect_setting(item_id)
        if cogmap_setting:
            cogmap_settings_data[cogmap_setting].append(item)
        
        # Add reasoning chain
        processed_item = reasoning_generator.add_reasoning_to_item(processed_item)
        
        # Track reasoning settings
        reasoning_setting_detected = reasoning_generator.detect_setting(item_id)
        if reasoning_setting_detected in reasoning_setting_counts:
            reasoning_setting_counts[reasoning_setting_detected] += 1
        
        processed_data.append(processed_item)
    
    # Print comprehensive statistics
    total_questions = len(processed_data)
    print(f"📊 Full pipeline processing summary:")
    print(f"Total number of questions processed: {total_questions}")
    
    # Cogmap statistics
    cogmap_counts = [
        f"{len(cogmap_settings_data['around'])} around questions",
        f"{len(cogmap_settings_data['among'])} among questions", 
        f"{len(cogmap_settings_data['translation'])} translation questions",
        f"{len(cogmap_settings_data['rotation'])} rotation questions"
    ]
    print(f"🧠 Cognitive maps generated: {', '.join(cogmap_counts)}")
    
    # Reasoning statistics
    print(f"🔗 Reasoning chains generated:")
    for setting_name, count in reasoning_setting_counts.items():
        if count > 0:
            print(f"  - {setting_name}: {count} items")
    
    # Save processed data
    try:
        save_jsonl(processed_data, output_path)
        print(f"✅ Full pipeline processing completed: {output_path}")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")


def process_data(input_path: str, output_path: Optional[str] = None, 
                task_type: TaskType = "cognitive_map",
                format_type: FormatType = "full",
                instruction_type: InstructionType = "both",
                suppress_warnings: bool = False,
                reasoning_setting: Optional[ReasoningType] = None) -> None:
    """
    Process data with format conversion and validation.
    Uses organized directory structure if no output path is specified.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (uses organized structure if None)
        task_type: Type of processing task
        format_type: Output format type
        instruction_type: Instruction type for cognitive maps
        suppress_warnings: Whether to suppress warnings
        reasoning_setting: Specific reasoning setting for reasoning tasks
    """
    # Use organized directory structure if no output path specified
    if output_path is None:
        output_path = get_default_scaffold_output_path(input_path, task_type)
        print(f"🔄 Processing data: {task_type} format={format_type}")
        print(f"📁 Using organized output path: {output_path}")
    else:
        print(f"🔄 Processing data: {task_type} format={format_type}")
        print(f"📁 Custom output path: {output_path}")
    
    if task_type == "cognitive_map":
        generate_cognitive_maps(input_path, output_path, format_type, instruction_type, suppress_warnings)
    elif task_type == "reasoning":
        generate_reasoning_chains(input_path, output_path, format_type, reasoning_setting)
    elif task_type == "full_pipeline":
        generate_full_pipeline(input_path, output_path, format_type, reasoning_setting)
    else:
        # For other task types, implement as needed
        print(f"⚠️ Task type '{task_type}' not yet implemented")


def batch_process(input_dir: str, output_dir: str, 
                 task_type: TaskType = "cognitive_map",
                 format_type: FormatType = "full",
                 instruction_type: InstructionType = "both",
                 suppress_warnings: bool = False,
                 reasoning_setting: Optional[ReasoningType] = None) -> None:
    """
    Batch process all JSONL files in a directory.
    
    Args:
        input_dir: Directory containing input JSONL files
        output_dir: Directory to save processed files
        task_type: Type of processing task
        format_type: Output format type
        instruction_type: Instruction type for cognitive maps
        suppress_warnings: Whether to suppress warnings
        reasoning_setting: Specific reasoning setting for reasoning tasks
    """
    print(f"📁 Batch processing: {input_dir} -> {output_dir}")
    
    # Find all JSONL files
    pattern = os.path.join(input_dir, "*.jsonl")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"❌ No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Process each file
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)
        
        print(f"\n🔧 Processing {base_name}...")
        
        try:
            process_data(input_file, output_file, task_type, format_type, 
                        instruction_type, suppress_warnings, reasoning_setting)
        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}")
    
    print(f"\n✅ Batch processing completed. Results saved to {output_dir}")


class CognitiveMapProcessor:
    """
    Processor specifically for cognitive mapping tasks.
    
    Provides a clean interface to the original cognitive map generation
    with additional validation and error handling.
    """
    
    def __init__(self, format_type: FormatType = "full"):
        """
        Initialize processor.
        
        Args:
            format_type: Cognitive map format
        """
        self.format_type = format_type
        self.generator = CogMapGenerator(format_type)
    
    def process_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process a single file and return statistics.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            
        Returns:
            Processing statistics
        """
        # Cast to FormatType for type safety
        format_type: FormatType = self.format_type  # type: ignore
        generate_cognitive_maps(input_path, output_path, format_type, "both", False)
        
        # Return basic statistics
        return {
            "input_file": input_path,
            "output_file": output_path,
            "format_type": self.format_type,
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
        required_fields = ["id", "question", "images"]
        
        for field in required_fields:
            if field not in item:
                return False
        
        # Check if setting can be detected
        item_id = item.get("id", "")
        setting = self.generator.detect_setting(item_id)
        
        return setting is not None


class ReasoningProcessor:
    """
    Processor specifically for reasoning chain generation tasks.
    
    Provides interface for reasoning chain generation with selective processing.
    """
    
    def __init__(self, reasoning_setting: Optional[ReasoningType] = None):
        """
        Initialize reasoning processor.
        
        Args:
            reasoning_setting: Specific reasoning setting to focus on
        """
        self.reasoning_setting = reasoning_setting
        self.generator = ReasoningChainGenerator(reasoning_setting)
    
    def process_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process a single file and return statistics.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            
        Returns:
            Processing statistics
        """
        # Cast to ReasoningType for type safety
        reasoning_setting: Optional[ReasoningType] = self.reasoning_setting  # type: ignore
        generate_reasoning_chains(input_path, output_path, "full", reasoning_setting)
        
        return {
            "input_file": input_path,
            "output_file": output_path,
            "reasoning_setting": self.reasoning_setting,
            "status": "completed"
        }
    
    def validate_item(self, item: Dict) -> bool:
        """
        Validate that an item can be processed for reasoning.
        
        Args:
            item: Data item to validate
            
        Returns:
            True if item is valid for reasoning processing
        """
        required_fields = ["id", "question", "gt_answer"]
        
        for field in required_fields:
            if field not in item:
                return False
        
        # Check if setting can be detected
        item_id = item.get("id", "")
        setting = self.generator.detect_setting(item_id)
        
        # If specific setting is required, check match
        if self.reasoning_setting:
            return setting == self.reasoning_setting
        
        return setting is not None


class DataProcessor:
    """
    General data processor for multiple task types.
    
    Provides a unified interface for different data processing tasks.
    """
    
    def __init__(self):
        """Initialize processor."""
        self.processors = {}
    
    def get_processor(self, task_type: TaskType, format_type: FormatType = "full", 
                     reasoning_setting: Optional[ReasoningType] = None):
        """
        Get appropriate processor for task type.
        
        Args:
            task_type: Type of processing task
            format_type: Output format type
            reasoning_setting: Reasoning setting for reasoning tasks
            
        Returns:
            Processor instance
        """
        if task_type == "cognitive_map":
            return CognitiveMapProcessor(format_type)
        elif task_type == "reasoning":
            return ReasoningProcessor(reasoning_setting)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def process(self, input_path: str, output_path: str,
                task_type: TaskType = "cognitive_map",
                format_type: FormatType = "full",
                reasoning_setting: Optional[ReasoningType] = None) -> Dict:
        """
        Process data using appropriate processor.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            task_type: Type of processing task
            format_type: Output format type
            reasoning_setting: Reasoning setting for reasoning tasks
            
        Returns:
            Processing results
        """
        processor = self.get_processor(task_type, format_type, reasoning_setting)
        return processor.process_file(input_path, output_path) 