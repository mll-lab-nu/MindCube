"""
MindCube Data Processing Framework

A unified data processing system for cognitive mapping tasks with minimal changes
to original code structure.

Key Features:
- Cognitive map generation (from original grounded_cogmap_gen.py) ✅ COMPLETED
- Reasoning chain generation (to be integrated from original reasoning scripts) 🚧 PENDING
- Data format conversion and validation  
- Batch processing capabilities
- Integration with existing data pipelines

Usage Examples:
    # Generate cognitive maps
    from src.scaffold_curation import generate_cognitive_maps
    generate_cognitive_maps("input.jsonl", "output.jsonl")
    
    # Generate both cognitive maps and reasoning chains (reasoning pending implementation)
    from src.scaffold_curation import generate_with_reasoning
    processed_item = generate_with_reasoning(data_item)
    
    # Process data in different formats
    from src.scaffold_curation import process_data
    process_data("input.jsonl", "output.jsonl", format_type="full")
    
    # Batch processing
    from src.scaffold_curation import batch_process
    batch_process("input_dir/", "output_dir/", task_type="cognitive_map")

Note on Reasoning Generation:
    The reasoning chain generation framework is in place but the actual generation
    logic is pending integration of the original reasoning scripts for each setting.
"""

# Import processors from root level
from .processors import (
    generate_cognitive_maps,
    process_data,
    batch_process,
    CognitiveMapProcessor,
    DataProcessor,
    generate_reasoning_chains,
    generate_full_pipeline
)

# Import cogmap-specific generators from cogmap subdirectory
from .cogmap.generators import (
    CogMapGenerator,
    ReasoningChainGenerator,
    generate_around_cogmap,
    generate_among_cogmap, 
    generate_translation_cogmap,
    generate_rotation_cogmap,
    generate_with_reasoning
)

__all__ = [
    # Main interfaces (recommended for most users)
    'generate_cognitive_maps',
    'generate_with_reasoning',  # Will be fully functional when reasoning scripts are integrated
    'process_data', 
    'batch_process',
    
    # Specific processors
    'CognitiveMapProcessor',
    'DataProcessor',
    
    # Generators (advanced users)
    'CogMapGenerator',
    'ReasoningChainGenerator',  # Framework ready, implementation pending
    'generate_around_cogmap',
    'generate_among_cogmap',
    'generate_translation_cogmap', 
    'generate_rotation_cogmap',
    'generate_reasoning_chains',
    'generate_full_pipeline'
]


def quick_start_guide():
    """Print a quick start guide for data processing."""
    print("""
🔧 MindCube Data Processing Framework - Quick Start

=== BASIC USAGE ===

1. Generate cognitive maps from raw data:
   ```python
   from src.scaffold_curation import generate_cognitive_maps
   generate_cognitive_maps("raw_data.jsonl", "processed_data.jsonl")
   ```

2. Process existing data with format conversion:
   ```python
   from src.scaffold_curation import process_data
   process_data("input.jsonl", "output.jsonl", format_type="full")
   ```

3. Batch processing for multiple files:
   ```python  
   from src.scaffold_curation import batch_process
   batch_process("input_dir/", "output_dir/", task_type="cognitive_map")
   ```

=== ADVANCED USAGE ===

4. Custom cognitive map generation:
   ```python
   from src.scaffold_curation import CogMapGenerator
   generator = CogMapGenerator(format_type="full")
   result = generator.generate_for_item(data_item)
   ```

5. Setting-specific generation:
   ```python
   from src.scaffold_curation import generate_around_cogmap
   cogmap, objects, oriented = generate_around_cogmap(item)
   ```

=== DATA FORMATS ===

Input: Raw CrossViewQA JSONL format
Output: Processed format with cognitive maps added

Supported cognitive map formats:
- "full": Complete objects + views format
- "shortened": Simplified format  
- "qwen": Format optimized for Qwen training
""") 