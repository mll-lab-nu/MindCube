#!/usr/bin/env python3
"""
MindCube Inference Script

Unified entry point for running inference with different models and backends.
Supports both open-source and closed-source models.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.open_source import OpenSourceInferenceEngine
from inference.closed_source import ClosedSourceInferenceEngine
from inference.utils import ConfigManager


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run inference with MindCube models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default HuggingFace model
  python scripts/run_inference.py --model-type qwen2.5vl \\
    --input-file data/prompts/general/test.jsonl --output-file results/output.jsonl

  # Specify HuggingFace model name
  python scripts/run_inference.py --model-type qwen2.5vl \\
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \\
    --input-file data/prompts/general/test.jsonl --output-file results/output.jsonl

  # Use local fine-tuned model
  python scripts/run_inference.py --model-type qwen2.5vl --model-path /path/to/local/model \\
    --input-file data/test.jsonl --output-file results/output.jsonl

  # Use vLLM acceleration (if available)
  python scripts/run_inference.py --model-type qwen2.5vl \\
    --backend vllm --input-file data/test.jsonl --output-file results/output.jsonl

  # Auto-generate output filename
  python scripts/run_inference.py --model-type qwen2.5vl \\
    --input-file data/test.jsonl --output-dir results/

  # Use configuration file
  python scripts/run_inference.py --config configs/qwen_inference.json
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="qwen2.5vl",
        help="Type of model to use (qwen2.5vl, gpt4v, etc.)"
    )
    parser.add_argument(
        "--model-path", 
        type=str,
        help="Path to the model or HuggingFace model name (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend (transformers or vllm)"
    )
    
    # Data configuration
    parser.add_argument(
        "--input-file", 
        type=str, 
        required=True,
        help="Path to input JSONL file with prompts and image paths"
    )
    
    # Make output more flexible
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output-file", 
        type=str,
        help="Path to output JSONL file for results"
    )
    output_group.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory (filename will be auto-generated based on input and model)"
    )
    
    parser.add_argument(
        "--image-root", 
        type=str, 
        default="./data/",
        help="Root directory for image files"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=4096,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=1.0,
        help="Top-p for nucleus sampling"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON configuration file"
    )
    
    # Other options
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for processing (default: 1, increase for better GPU utilization)"
    )
    parser.add_argument(
        "--multi-gpu", 
        action="store_true",
        help="Use multiple GPUs for balanced load (default: single GPU)"
    )

    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List supported models and exit"
    )
    
    return parser


def generate_output_filename(input_file: str, model_type: str, model_path: str = None) -> str:
    """
    Generate output filename based on input file and model info.
    
    Args:
        input_file: Path to input file
        model_type: Type of model
        model_path: Path to model (for extracting model name)
        
    Returns:
        Generated output filename
    """
    input_path = Path(input_file)
    input_stem = input_path.stem  # filename without extension
    
    # Extract model identifier
    if model_path and "/" in model_path:
        # HuggingFace model name like "Qwen/Qwen2.5-VL-3B-Instruct"
        model_id = model_path.split("/")[-1].lower()
    elif model_path and os.path.exists(model_path):
        # Local path - use directory name
        model_id = Path(model_path).name.lower()
    else:
        # Use model type
        model_id = model_type.lower()
    
    # Generate output filename
    output_filename = f"{input_stem}_{model_id}_responses.jsonl"
    return output_filename


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}


def create_inference_engine(args: argparse.Namespace) -> Any:
    """Create appropriate inference engine based on arguments."""
    model_type = args.model_type.lower()
    
    # Check if it's a supported open source model
    open_source_models = OpenSourceInferenceEngine.list_supported_models()
    closed_source_models = ClosedSourceInferenceEngine.list_supported_models()
    
    if model_type in open_source_models:
        if not args.model_path:
            # Set default HuggingFace model if no path specified
            if model_type in ['qwen2.5vl', 'qwen', 'qwen2.5-vl']:
                args.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
                print(f"Using default HuggingFace model: {args.model_path}")
            else:
                raise ValueError("--model-path is required for open source models")
        
        # Create engine with configuration
        kwargs = {
            'backend': args.backend,
            'max_new_tokens': args.max_new_tokens,
            'generation_config': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'do_sample': args.temperature > 0
            }
        }
        
        # Add all config file parameters if they exist
        if hasattr(args, 'vllm_config') and args.vllm_config:
            kwargs.update(args.vllm_config)
            print(f"Using vLLM config: {args.vllm_config}")
        
        # Add other config parameters
        for attr in ['gpu_memory_utilization', 'max_model_len', 'tensor_parallel_size', 
                     'limit_mm_per_prompt', 'trust_remote_code', 'dtype', 'enable_prefix_caching',
                     'enable_chunked_prefill', 'max_num_seqs', 'max_num_batched_tokens', 'block_size']:
            if hasattr(args, attr):
                kwargs[attr] = getattr(args, attr)
        
        # Add GPU control options - default is single GPU
        if args.multi_gpu:
            kwargs['single_gpu'] = False  # Enable multi-GPU
        else:
            kwargs['single_gpu'] = True   # Default: single GPU
        
        return OpenSourceInferenceEngine.create_engine(
            model_type, args.model_path, **kwargs
        )
        
    elif model_type in closed_source_models:
        # Placeholder for closed source models
        return ClosedSourceInferenceEngine(model_type)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def validate_input_file(input_file: str) -> bool:
    """Validate input file format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return False
    
    try:
        with open(input_file, 'r') as f:
            first_line = f.readline()
            if not first_line.strip():
                print("Error: Input file is empty")
                return False
            
            data = json.loads(first_line.strip())
            required_fields = ['input_prompt']  # 'images' is optional
            for field in required_fields:
                if field not in data:
                    print(f"Error: Required field '{field}' not found in input file")
                    return False
                    
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return False
    except Exception as e:
        print(f"Error validating input file: {e}")
        return False


def list_models():
    """List all supported models."""
    print("Supported Open Source Models:")
    for model in OpenSourceInferenceEngine.list_supported_models():
        info = OpenSourceInferenceEngine.get_model_info(model)
        print(f"  - {model}: {info.get('name', 'Unknown')} - {info.get('description', 'No description')}")
    
    print("\nSupported Closed Source Models:")
    for model in ClosedSourceInferenceEngine.list_supported_models():
        print(f"  - {model}")


def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        list_models()
        return
    
    # Load configuration file if provided
    if args.config:
        config = load_config_from_file(args.config)
        print(f"Loaded config from {args.config}: {config}")
        
        # Simple approach: Only set config values if not explicitly provided on command line
        # We'll track which arguments were explicitly set by checking sys.argv
        import sys
        explicit_args = set()
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--'):
                arg_name = arg[2:].replace('-', '_')
                explicit_args.add(arg_name)
        
        # Update args with config values
        for key, value in config.items():
            if key == 'vllm_config':
                # Handle nested vllm_config specially
                setattr(args, key, value)
                # Also set individual vllm parameters at top level for easy access
                for vllm_key, vllm_value in value.items():
                    if vllm_key not in explicit_args:
                        setattr(args, vllm_key, vllm_value)
                        print(f"  Using config value for {vllm_key}: {vllm_value}")
                    else:
                        print(f"  Command line value for {vllm_key} takes precedence")
            elif key == 'generation_config':
                # Handle generation_config specially
                gen_config = value
                for gen_key, gen_value in gen_config.items():
                    if gen_key not in explicit_args:
                        setattr(args, gen_key, gen_value)
                        print(f"  Using config value for {gen_key}: {gen_value}")
                    else:
                        print(f"  Command line value for {gen_key} takes precedence")
            else:
                # Handle regular parameters
                if key not in explicit_args:
                    setattr(args, key, value)
                    print(f"  Using config value for {key}: {value}")
                else:
                    print(f"  Command line value for {key} takes precedence: {getattr(args, key)}")
    
    # Validate required arguments
    if not args.input_file:
        parser.error("--input-file is required")
    
    # Handle output file generation
    if args.output_dir:
        if not args.model_path:
            # Need to determine model path first for filename generation
            if args.model_type.lower() in ['qwen2.5vl', 'qwen', 'qwen2.5-vl']:
                temp_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            else:
                temp_model_path = args.model_type
        else:
            temp_model_path = args.model_path
            
        output_filename = generate_output_filename(
            args.input_file, args.model_type, temp_model_path
        )
        args.output_file = os.path.join(args.output_dir, output_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Auto-generated output file: {args.output_file}")
    
    # Validate input file
    if not validate_input_file(args.input_file):
        sys.exit(1)
    
    if args.verbose:
        print(f"Model type: {args.model_type}")
        print(f"Model path: {args.model_path}")
        print(f"Backend: {args.backend}")
        print(f"Input file: {args.input_file}")
        print(f"Output file: {args.output_file}")
        print(f"Image root: {args.image_root}")
    
    try:
        # Create inference engine
        if args.verbose:
            print("Creating inference engine...")
        engine = create_inference_engine(args)
        
        # Run batch inference
        if args.verbose:
            print("Starting inference...")
        engine.batch_infer(
            data_file=args.input_file,
            output_file=args.output_file,
            image_root=args.image_root,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"Inference completed successfully! Results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 