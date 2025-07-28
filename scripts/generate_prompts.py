#!/usr/bin/env python3
"""
MindCube Prompt Generation Script

Generate final model-ready prompts from scaffold data for different task settings.
Supports multiple task types with automatic detection and batch processing.

Usage:
    # Auto-detect task types and generate prompts
    python generate_prompts.py --input scaffold_data.jsonl --output prompts.jsonl
    
    # Generate prompts for specific task type
    python generate_prompts.py --input scaffold_data.jsonl --output cogmap_prompts.jsonl --task cogmap_qa
    
    # Batch processing
    python generate_prompts.py --batch_dir scaffold_data/ --output_dir prompts/
    
    # Generate all task types separately
    python generate_prompts.py --input scaffold_data.jsonl --output_dir prompts/ --all_tasks
    
    # Show help
    python generate_prompts.py --help
"""

import sys
import os
import argparse
from typing import cast

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.prompt_generation import (
    generate_task_prompts,
    batch_generate_prompts,
    PromptProcessor
)
from src.prompt_generation.processors import (
    validate_scaffold_data,
    quick_prompt_sample,
    generate_all_task_prompts,
    get_default_prompt_output_dir
)
from src.utils import ensure_dir
from src.prompt_generation.generators import list_task_types, TaskType


def main():
    """Main entry point for prompt generation."""
    parser = argparse.ArgumentParser(
        description='MindCube Prompt Generation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect task types and generate prompts
  python generate_prompts.py --input scaffold_data.jsonl --output prompts.jsonl
  
  # Generate prompts for specific task type
  python generate_prompts.py --input scaffold_data.jsonl --output cogmap_prompts.jsonl --task cogmap_qa
  
  # Generate reasoning QA prompts
  python generate_prompts.py --input scaffold_data.jsonl --output reasoning_prompts.jsonl --task reasoning_qa
  
  # Generate full pipeline prompts
  python generate_prompts.py --input scaffold_data.jsonl --output full_prompts.jsonl --task full_pipeline
  
  # Batch processing (auto-detect)
  python generate_prompts.py --batch_dir scaffold_data/ --output_dir prompts/
  
  # Generate all task types separately
  python generate_prompts.py --input scaffold_data.jsonl --output_dir prompts/ --all_tasks
  
  # Validate scaffold data
  python generate_prompts.py --input scaffold_data.jsonl --validate
  
  # Quick preview
  python generate_prompts.py --input scaffold_data.jsonl --preview --task reasoning_qa
        """
    )
    
    # Main action arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str,
                       help='Input JSONL file with scaffold data')
    group.add_argument('--batch_dir', '-b', type=str,
                       help='Directory containing scaffold JSONL files for batch processing')
    group.add_argument('--list_tasks', action='store_true',
                       help='List available task types')
    
    # Task-specific arguments
    parser.add_argument('--task', '-t', 
                       choices=[
                           'raw_qa', 'ff_rsn', 
                           'aug_cgmap_in', 'aug_cgmap_out', 'plain_cgmap_out',
                           'plain_cgmap_ffr_out', 'aug_cgmap_ffr_out', 'cgmap_in_ffr_out', 
                           'cgmap_in_cgmap_out',
                           'noformat_ff_rsn', 'nl_cgmap_in_ffr_out', 'nl_aug_cgmap_ffr_out', 'nl_plain_cgmap_ffr_out'
                       ],
                       default=None,
                       help='Specific task type to generate (auto-detected if not specified)')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (for single file processing)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (for batch processing or all_tasks mode)')
    
    # Generation modes
    parser.add_argument('--all_tasks', action='store_true',
                       help='Generate separate files for all task types')
    parser.add_argument('--auto_detect', action='store_true', default=True,
                       help='Auto-detect task types for each item (default)')
    
    # Utility functions
    parser.add_argument('--validate', action='store_true',
                       help='Validate scaffold data and show statistics')
    parser.add_argument('--preview', action='store_true',
                       help='Generate preview samples without full processing')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples for preview mode (default: 3)')
    
    # Processing options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # List task types and exit
    if args.list_tasks:
        print("💬 Available Task Types:")
        task_descriptions = {
            "raw_qa": "Raw QA without cognitive maps or reasoning chains",
            "ff_rsn": "Free form reasoning",
            "aug_cgmap_in": "Augmented cognitive map as input -> Direct answer",
            "aug_cgmap_out": "Raw views + question -> Direct answer",
            "plain_cgmap_out": "Plain cognitive map -> Direct answer", 
            "plain_cgmap_ffr_out": "Raw views + question -> Plain map + reasoning -> answer",
            "aug_cgmap_ffr_out": "Raw views + question -> Aug. map + reasoning -> answer",
            "cgmap_in_ffr_out": "Augmented cognitive map as input -> Reasoning -> answer",
            "cgmap_in_cgmap_out": "Augmented cognitive map as input -> Reasoning -> answer"
        }
        
        for task_type in list_task_types():
            description = task_descriptions.get(task_type, "No description available")
            print(f"  ✅ {task_type}: {description}")
        
        print("\nUsage:")
        print("  python generate_prompts.py --input data.jsonl --output output.jsonl --task reasoning_qa")
        return
    
    try:
        if args.validate:
            # Validation mode
            if not args.input:
                print("❌ Error: --input required for validation mode")
                sys.exit(1)
            
            if not args.quiet:
                print(f"🔍 Validating scaffold data: {args.input}")
            
            report = validate_scaffold_data(args.input)
            
            if report["status"] == "error":
                print(f"❌ Validation failed: {report['error']}")
                sys.exit(1)
            
            # Print detailed report
            print(f"\n📊 Validation Report:")
            print(f"  Total items: {report['total_items']}")
            
            print(f"\n  Task type distribution:")
            for task_type, count in report['task_type_distribution'].items():
                if count > 0:
                    print(f"    - {task_type}: {count} items")
            
            if report['recommendations']:
                print(f"\n  💡 Recommendations:")
                for rec in report['recommendations']:
                    print(f"    - {rec}")
            
            return
        
        elif args.preview:
            # Preview mode
            if not args.input:
                print("❌ Error: --input required for preview mode")
                sys.exit(1)
            
            if not args.quiet:
                print(f"📝 Generating prompt preview: {args.input}")
            
            # Cast task to TaskType for type safety
            task_type = cast(TaskType, args.task) if args.task else None
            quick_prompt_sample(args.input, task_type, args.samples)
            return
        
        elif args.batch_dir:
            # Batch processing
            output_dir = args.output_dir or f"{args.batch_dir}_prompts"
            
            if not os.path.exists(args.batch_dir):
                print(f"❌ Error: Batch directory '{args.batch_dir}' not found")
                sys.exit(1)
            
            if not args.quiet:
                print(f"📁 Batch prompt generation: {args.batch_dir} -> {output_dir}")
            
            if args.task:
                # Specific task type
                task_type = cast(TaskType, args.task)
                batch_generate_prompts(args.batch_dir, output_dir, [task_type], auto_detect=False)
            else:
                # Auto-detect mode
                batch_generate_prompts(args.batch_dir, output_dir, auto_detect=True)
        
        elif args.input:
            # Single file processing
            if not os.path.exists(args.input):
                print(f"❌ Error: Input file '{args.input}' not found")
                sys.exit(1)
            
            if args.all_tasks:
                # Generate all task types separately using new organized structure
                output_dir = args.output_dir  # Use custom dir if specified, otherwise use default
                
                if not args.quiet:
                    if output_dir:
                        print(f"🔄 Generating ALL task types from: {args.input}")
                        print(f"📁 Custom output directory: {output_dir}")
                    else:
                        print(f"🔄 Generating ALL task types from: {args.input}")
                        print(f"📁 Using default directory: ./data/prompts/general/")
                
                generate_all_task_prompts(args.input, output_dir)
            
            else:
                # Single task type or auto-detect
                if not args.output:
                    # Use default prompt directory for all outputs
                    default_output_dir = get_default_prompt_output_dir()
                    ensure_dir(default_output_dir)
                    
                    base_name = os.path.splitext(os.path.basename(args.input))[0]
                    if args.task:
                        args.output = os.path.join(default_output_dir, f"{base_name}_{args.task}.jsonl")
                    else:
                        args.output = os.path.join(default_output_dir, f"{base_name}.jsonl")
                
                if not args.quiet:
                    task_desc = f" ({args.task})" if args.task else " (auto-detect)"
                    print(f"💬 Generating prompts{task_desc}: {args.input} -> {args.output}")
                
                # Cast task to TaskType for type safety
                task_type = cast(TaskType, args.task) if args.task else None
                generate_task_prompts(args.input, args.output, task_type, auto_detect=(args.task is None))
                
                if not args.quiet:
                    print(f"✅ Prompt generation completed: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 