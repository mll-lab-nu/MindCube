#!/usr/bin/env python3
"""
Command-line interface for MindCube evaluation framework.

This script provides easy command-line access to the evaluation functions.
"""

import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import evaluate, auto_evaluate, batch_evaluate, quick_start_guide


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MindCube Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python cli.py --input responses.jsonl --task basic
  
  # Cognitive map evaluation
  python cli.py --input responses.jsonl --task cogmap
  
  # Auto-detect task type
  python cli.py --input responses.jsonl --auto
  
  # Batch evaluation
  python cli.py --batch_dir results/ --output_dir analysis/
  
  # Show usage guide
  python cli.py --guide
        """
    )
    
    # Main action arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str,
                       help='Input JSONL file path')
    group.add_argument('--batch_dir', '-b', type=str,
                       help='Directory containing JSONL files for batch evaluation')
    group.add_argument('--guide', '-g', action='store_true',
                       help='Show quick start guide')
    
    # Task type arguments
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--task', '-t', choices=['basic', 'cogmap', 'cognitive_map'],
                           default='basic',
                           help='Task type: basic (answer accuracy) or cogmap (full evaluation)')
    task_group.add_argument('--auto', '-a', action='store_true',
                           help='Auto-detect task type from content')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path for results')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for batch evaluation')
    
    # Evaluation options
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation (basic metrics only for cogmap tasks)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # Show guide and exit
    if args.guide:
        quick_start_guide()
        return
    
    try:
        if args.batch_dir:
            # Batch evaluation
            if not args.quiet:
                print(f"🔄 Running batch evaluation on {args.batch_dir}")
            batch_evaluate(args.batch_dir, args.output_dir)
            
        elif args.input:
            # Single file evaluation
            if not os.path.exists(args.input):
                print(f"❌ Error: Input file '{args.input}' not found")
                sys.exit(1)
            
            if args.auto:
                # Auto-detect task type
                if not args.quiet:
                    print(f"🔍 Auto-detecting task type for {args.input}")
                results = auto_evaluate(args.input, args.output)
            else:
                # Use specified task type
                if not args.quiet:
                    print(f"📊 Running {args.task} evaluation on {args.input}")
                
                # Handle quick mode for cognitive map tasks
                if args.task in ['cogmap', 'cognitive_map'] and args.quick:
                    results = evaluate(args.input, args.task, args.output, 
                                     include_detailed_metrics=False)
                else:
                    results = evaluate(args.input, args.task, args.output)
            
            # Print summary if not quiet
            if not args.quiet:
                accuracy = results['results']['gen_cogmap_accuracy'] * 100
                total = results['results']['total']
                print(f"\n✅ Evaluation completed: {accuracy:.1f}% accuracy ({total} examples)")
                
                if args.output:
                    print(f"📁 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 