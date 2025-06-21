#!/usr/bin/env python3
"""
MindCube Scaffold Data Processing Script

Main entry point for MindCube scaffold data generation pipeline.
Supports cognitive map generation, reasoning chain creation, and full pipeline processing.

This script handles the SCAFFOLD GENERATION phase only.
For PROMPT GENERATION, use generate_prompts.py instead.

Usage:
    # Generate cognitive maps only
    python data_processing.py --input data.jsonl --task cogmap
    
    # Generate reasoning chains only  
    python data_processing.py --input data.jsonl --task reasoning --reasoning-setting rotation
    
    # Generate both cognitive maps and reasoning chains (default)
    python data_processing.py --input data.jsonl --task full_pipeline --reasoning-setting rotation
    
    # Custom output path (otherwise uses organized directory structure)
    python data_processing.py --input data.jsonl --output custom_path.jsonl --task cogmap
"""

import sys
import os
import argparse
from typing import cast

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.scaffold_curation.processors import process_data, batch_process, TaskType as ScaffoldTaskType


def main():
    """Main entry point for data processing."""
    parser = argparse.ArgumentParser(
        description='MindCube Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scaffold Generation (with organized directory structure):
  python data_processing.py --input data.jsonl --task cogmap
  python data_processing.py --input data.jsonl --task reasoning --reasoning-setting rotation
  python data_processing.py --input data.jsonl --task full_pipeline --reasoning-setting rotation
  
  # Custom output path:
  python data_processing.py --input data.jsonl --output custom.jsonl --task cogmap
  
  # Batch Processing:
  python data_processing.py --batch_dir input_dir/ --output_dir output_dir/ --task cogmap
  
  # For PROMPT generation, use generate_prompts.py instead!
        """
    )
    
    # Main arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str,
                       help='Input JSONL file')
    group.add_argument('--batch_dir', '-b', type=str,
                       help='Directory containing input JSONL files for batch processing')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (for single file processing)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (for batch processing)')
    
    # Task selection
    parser.add_argument('--task', '-t', 
                       choices=['cogmap', 'reasoning', 'full_pipeline'],
                       default='full_pipeline',
                       help='Scaffold processing task type')
    
    # Scaffold generation options
    parser.add_argument('--reasoning-setting', 
                       choices=['rotation', 'translation', 'among', 'around'],
                       help='Reasoning setting for reasoning chain generation')
    

    
    # Processing options
    parser.add_argument('--format', 
                       choices=['full', 'shortened', 'qwen'],
                       default='full',
                       help='Output format for scaffold generation (default: full)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    try:
            # Scaffold generation mode - map command line task names to internal task types
            task_mapping = {
                'cogmap': 'cognitive_map',
                'reasoning': 'reasoning', 
                'full_pipeline': 'full_pipeline'
            }
            internal_task_type = cast(ScaffoldTaskType, task_mapping.get(args.task, args.task))
            
            if args.batch_dir:
                # Batch processing
                if not args.output_dir:
                    args.output_dir = f"{args.batch_dir}_processed"
                
                if not args.quiet:
                    print(f"📁 Batch processing: {args.batch_dir} -> {args.output_dir}")
                
                batch_process(args.batch_dir, args.output_dir, internal_task_type, args.format, "both", False, args.reasoning_setting)
            
            else:
                # Single file processing
                if not args.input:
                    print("❌ Error: --input required for scaffold generation")
                    sys.exit(1)
                
                # Use organized directory structure if no output specified
                output_path = args.output  # Will be None if not specified
                
                if not args.quiet:
                    if output_path:
                        print(f"🔧 Processing: {args.input} -> {output_path}")
                    else:
                        print(f"🔧 Processing with organized directory structure: {args.input}")
                    print(f"   Task: {args.task}")
                    if args.reasoning_setting:
                        print(f"   Reasoning setting: {args.reasoning_setting}")
                
                process_data(args.input, output_path, internal_task_type, args.format, "both", False, args.reasoning_setting)
                
                if not args.quiet:
                    if output_path:
                        print(f"✅ Processing completed: {output_path}")
                    else:
                        print(f"✅ Processing completed with organized structure")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 