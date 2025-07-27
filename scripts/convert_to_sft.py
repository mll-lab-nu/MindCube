#!/usr/bin/env python3
"""
MindCube SFT Data Conversion Script

Converts MindCube general prompt data to model-specific SFT training formats.
Supports multiple model types with extensible architecture.

Usage:
    # Convert single file to Qwen format
    python scripts/convert_to_sft.py --input data/prompts/general/MindCube_tinybench_raw_qa.jsonl --output data/prompts/training/qwen2.5vl/raw_qa_sft.json --model qwen2.5vl
    
    # Batch convert all files in directory
    python scripts/convert_to_sft.py --input_dir data/prompts/general/ --output_dir data/prompts/training/qwen2.5vl/ --model qwen2.5vl
    
    # List supported models
    python scripts/convert_to_sft.py --list_models
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.data_formatters import (
    convert_prompts_to_sft_format,
    batch_convert_prompts_to_sft,
    list_supported_models,
    ModelType
)


def get_default_sft_output_dir(model_type: str) -> str:
    """Get default output directory for SFT data."""
    return f"./data/prompts/training/{model_type}/"


def main():
    parser = argparse.ArgumentParser(
        description="Convert MindCube prompt data to model-specific SFT formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file to Qwen format:
  python scripts/convert_to_sft.py --input data/prompts/general/MindCube_tinybench_raw_qa.jsonl --model qwen2.5vl
  
  # Convert with custom output:
  python scripts/convert_to_sft.py --input data/prompts/general/MindCube_tinybench_raw_qa.jsonl --output data/prompts/training/my_qwen_data.json --model qwen2.5vl
  
  # Batch convert all files:
  python scripts/convert_to_sft.py --input_dir data/prompts/general/ --model qwen2.5vl
  
  # Custom batch conversion:
  python scripts/convert_to_sft.py --input_dir data/prompts/general/ --output_dir data/prompts/training/custom/ --model qwen2.5vl
  
  # List available models:
  python scripts/convert_to_sft.py --list_models
        """
    )
    
    # Input/output arguments
    parser.add_argument('--input', '-i',
                        help='Input JSONL file with prompt data')
    parser.add_argument('--output', '-o',
                        help='Output JSON file for SFT training (auto-generated if not specified)')
    parser.add_argument('--input_dir', 
                        help='Input directory containing prompt JSONL files (for batch conversion)')
    parser.add_argument('--output_dir',
                        help='Output directory for SFT JSON files (auto-generated if not specified)')
    
    # Model configuration
    parser.add_argument('--model', '-m',
                        choices=['qwen2.5vl', 'llava', 'instructblip', 'internvl'],
                        help='Target model type for SFT format')
    
    # Utility arguments
    parser.add_argument('--list_models', action='store_true',
                        help='List supported model types')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        print("📋 Supported model types:")
        for model_type in list_supported_models():
            print(f"  - {model_type}")
        return
    
    # Validate arguments
    if not args.input and not args.input_dir:
        print("❌ Error: Must specify either --input or --input_dir")
        parser.print_help()
        sys.exit(1)
    
    if not args.model:
        print("❌ Error: Must specify --model")
        parser.print_help()
        sys.exit(1)
    
    if args.input and args.input_dir:
        print("❌ Error: Cannot specify both --input and --input_dir")
        sys.exit(1)
    
    try:
        if args.input_dir:
            # Batch conversion mode
            if not args.output_dir:
                args.output_dir = get_default_sft_output_dir(args.model)
            
            if not args.quiet:
                print(f"🔄 Batch converting prompt files to {args.model} SFT format...")
                print(f"📁 Input directory: {args.input_dir}")
                print(f"📁 Output directory: {args.output_dir}")
            
            batch_convert_prompts_to_sft(args.input_dir, args.output_dir, args.model)
            
            if not args.quiet:
                print(f"✅ Batch conversion completed: {args.output_dir}")
        
        else:
            # Single file conversion mode
            if not args.output:
                # Generate default output filename
                from src.training.data_formatters import get_formatter
                formatter = get_formatter(args.model)
                base_name = os.path.basename(args.input)
                output_filename = formatter.get_output_filename(base_name)
                default_output_dir = get_default_sft_output_dir(args.model)
                args.output = os.path.join(default_output_dir, output_filename)
            
            if not args.quiet:
                print(f"🔄 Converting prompt file to {args.model} SFT format...")
            
            convert_prompts_to_sft_format(args.input, args.output, args.model)
            
            if not args.quiet:
                print(f"✅ Conversion completed: {args.output}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 