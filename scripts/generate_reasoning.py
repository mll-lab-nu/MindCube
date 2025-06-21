#!/usr/bin/env python3
"""
MindCube Reasoning Chain Generation Script

Script specifically for generating reasoning chains with support for selective processing
by reasoning setting (around, among, translation, rotation).

Usage:
    # Generate all reasoning chains
    python generate_reasoning.py --input data.jsonl --output data_with_reasoning.jsonl
    
    # Generate reasoning chains for specific setting
    python generate_reasoning.py --input data.jsonl --output translation_reasoning.jsonl --setting translation
    
    # Batch processing
    python generate_reasoning.py --batch_dir raw_data/ --output_dir reasoning_data/ --setting among
    
    # Show help
    python generate_reasoning.py --help
"""

import sys
import os
import argparse

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.scaffold_curation.reasoning.generators import (
    batch_generate_reasoning_chains,
    ReasoningChainGenerator
)
from src.scaffold_curation.processors import batch_process, process_data


def main():
    """Main entry point for reasoning chain generation."""
    parser = argparse.ArgumentParser(
        description='MindCube Reasoning Chain Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all reasoning chains
  python generate_reasoning.py --input crossviewQA.jsonl --output crossviewQA_with_reasoning.jsonl
  
  # Generate only translation reasoning chains
  python generate_reasoning.py --input crossviewQA.jsonl --output translation_reasoning.jsonl --setting translation
  
  # Generate only rotation reasoning chains
  python generate_reasoning.py --input crossviewQA.jsonl --output rotation_reasoning.jsonl --setting rotation
  
  # Batch processing for specific setting
  python generate_reasoning.py --batch_dir raw_data/ --output_dir reasoning_data/ --setting among
  
  # Show available settings
  python generate_reasoning.py --list-settings
        """
    )
    
    # Main action arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str,
                       help='Input JSONL file path')
    group.add_argument('--batch_dir', '-b', type=str,
                       help='Directory containing JSONL files for batch processing')
    group.add_argument('--list-settings', action='store_true',
                       help='List available reasoning settings')
    
    # Reasoning-specific arguments
    parser.add_argument('--setting', '-s', 
                       choices=['around', 'among', 'translation', 'rotation'],
                       default=None,
                       help='Specific reasoning setting to process (if not specified, processes all)')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for batch processing')
    
    # Processing options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    # List available settings and exit
    if args.list_settings:
        print("🧠 Available Reasoning Settings:")
        settings_info = {
            "around": "Around spatial reasoning (placeholder - awaiting data)",
            "among": "Among spatial reasoning scenarios",  
            "translation": "Translation/linear spatial reasoning",
            "rotation": "Rotation spatial reasoning"
        }
        
        for setting, description in settings_info.items():
            status = "⚠️" if setting == "around" else "✅"
            print(f"  {status} {setting}: {description}")
        
        print("\nUsage:")
        print("  python generate_reasoning.py --input data.jsonl --output output.jsonl --setting translation")
        return
    
    try:
        if args.batch_dir:
            # Batch processing
            output_dir = args.output_dir or f"{args.batch_dir}_reasoning"
            
            if not os.path.exists(args.batch_dir):
                print(f"❌ Error: Batch directory '{args.batch_dir}' not found")
                sys.exit(1)
            
            if not args.quiet:
                setting_desc = f" (setting: {args.setting})" if args.setting else " (all settings)"
                print(f"🧠 Running batch reasoning generation on {args.batch_dir}{setting_desc}")
            
            if args.dry_run:
                print("🔍 Dry run mode - would process:")
                import glob
                pattern = os.path.join(args.batch_dir, "*.jsonl")
                files = glob.glob(pattern)
                for file in files:
                    print(f"  - {file}")
                print(f"Output directory: {output_dir}")
                return
            
            # Use batch_process with reasoning task
            batch_process(args.batch_dir, output_dir, "reasoning", "full", "both", False, args.setting)
            
        elif args.input:
            # Single file processing
            if not os.path.exists(args.input):
                print(f"❌ Error: Input file '{args.input}' not found")
                sys.exit(1)
            
            # Generate output path if not provided
            if not args.output:
                base_name = os.path.splitext(args.input)[0]
                setting_suffix = f"_{args.setting}" if args.setting else "_reasoning"
                args.output = f"{base_name}{setting_suffix}.jsonl"
            
            if not args.quiet:
                setting_desc = f" (setting: {args.setting})" if args.setting else " (all settings)"
                print(f"🧠 Generating reasoning chains: {args.input} -> {args.output}{setting_desc}")
            
            if args.dry_run:
                print("🔍 Dry run mode - would process:")
                print(f"  Input: {args.input}")
                print(f"  Output: {args.output}")
                print(f"  Setting: {args.setting or 'all'}")
                
                # Count items by setting
                import json
                generator = ReasoningChainGenerator()
                with open(args.input, 'r', encoding='utf-8') as f:
                    items = [json.loads(line) for line in f if line.strip()]
                
                setting_counts = {"around": 0, "among": 0, "translation": 0, "rotation": 0}
                for item in items:
                    item_setting = generator.detect_setting(item.get("id", ""))
                    if item_setting in setting_counts:
                        setting_counts[item_setting] += 1
                
                print(f"  Items to process:")
                for setting_name, count in setting_counts.items():
                    if args.setting and setting_name != args.setting:
                        continue
                    if count > 0:
                        print(f"    - {setting_name}: {count} items")
                return
            
            # Process the file
            batch_generate_reasoning_chains(args.input, args.output, args.setting)
            
            if not args.quiet:
                print(f"✅ Reasoning chain generation completed: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 